"""
Integrated Tract-Level Data Collection Script

This script fetches and integrates data from multiple sources at the Census Tract level:
1. American Community Survey 5-year estimates (ACS5) from the Census Bureau
2. Geographic boundaries from TIGER/Line (via tigris)
3. PLACES health data from CDC

All datasets are joined using the GEOID identifier common across these sources.
"""

import os
import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import json
from census import Census
from us import states
import pygris
import time
import logging
import sys
import random
from datetime import datetime
from typing import List, Dict, Union, Optional, Tuple, Any, Set, cast

# Import NLTK for random word generation
try:
    import nltk
    from nltk.corpus import words, brown
    # Download required NLTK data if not already present
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words', quiet=True)
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        nltk.download('brown', quiet=True)
    
    # Create word lists for token generation
    nltk_words = words.words()
    brown_words = list(set(word.lower() for word in brown.words() if len(word) > 3 and word.isalpha()))
    combined_words = list(set(nltk_words + brown_words))
    
    def generate_random_token() -> str:
        """Generate a random token using 3-7 words from NLTK corpora."""
        num_words = random.randint(3, 7)
        selected_words = [random.choice(combined_words) for _ in range(num_words)]
        # Filter out any non-alphabetic words and ensure they're lowercase
        selected_words = [word.lower() for word in selected_words if word.isalpha()]
        # Join with underscores
        return "_".join(selected_words)
        
except ImportError:
    # Fallback if NLTK is not available
    print("NLTK not available, using simple random token generation")
    
    def generate_random_token() -> str:
        """Generate a random token without NLTK."""
        # Simple fallback using random characters
        word_count = random.randint(3, 7)
        words = []
        for _ in range(word_count):
            word_length = random.randint(4, 10)
            word = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(word_length))
            words.append(word)
        return "_".join(words)

# Setup logging system with real-time streaming
def setup_logger(name, log_file, level=logging.INFO, stream_to_console=True):
    """Function to setup a logger that writes to both file and console in real-time"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Create full path for log file
    log_path = os.path.join(log_dir, log_file)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Don't propagate to parent loggers
    
    # Remove existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Create formatter with timestamps
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler for real-time updates
    file_handler = logging.FileHandler(log_path, mode='a')
    file_handler.setFormatter(formatter)
    # Don't set flush attribute directly, it's a method not a property
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if stream_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# Create main logger
main_logger = setup_logger('integrated_tract_data', 'main.log')

# Create individual loggers for each data source
acs_logger = setup_logger('acs_data', 'acs_data.log')
tigris_logger = setup_logger('tigris_data', 'tigris_data.log')
places_logger = setup_logger('places_data', 'places_data.log')
integration_logger = setup_logger('integration', 'integration.log')

# Get API Keys from various sources with fallbacks
import os

# Try to load from .env file first
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file if it exists
    load_dotenv()
    print("Attempted to load .env file")
except ImportError:
    print("python-dotenv not installed, skipping .env file loading")

# Try to import from config file, then fall back to environment variables
try:
    from config import CENSUS_API_KEY
    print("Using Census API key from config.py")
except ImportError:
    # If config.py doesn't exist or doesn't contain the keys, use environment variables
    CENSUS_API_KEY: str = os.environ.get("CENSUS_API_KEY", "")  # Get from environment variable
    print("Using Census API key from environment variables")
    
# Verify if Census API key is available
if not CENSUS_API_KEY:
    print("WARNING: No Census API key found. Some functionality will be limited.")

# Define the current year for most recent data
CURRENT_YEAR: int = datetime.now().year
# ACS5 data is typically 2 years behind
ACS_YEAR: int = 2022 if CURRENT_YEAR >= 2022 else 2020

class IntegratedTractData:
    """Class to fetch and integrate tract-level data from multiple sources."""
    
    def __init__(self, census_api_key: str = CENSUS_API_KEY, 
             state_fips: Optional[List[str]] = None, county_fips: Optional[List[str]] = None, 
             output_path: Optional[str] = None, log_level: int = logging.INFO) -> None:
        """
        Initialize the data fetcher with API keys and optional geographic filters.
        
        Parameters:
        -----------
        census_api_key : str
            Census Bureau API key
        state_fips : list or None
            List of state FIPS codes to filter data (None for all states)
        county_fips : list or None
            List of county FIPS codes to filter data (None for all counties)
        output_path : str or None
            Path to save output files (None for no saving)
        log_level : int
            Logging level (default: logging.INFO)
        """
        self.census_api_key: str = census_api_key
        self.state_fips: Optional[List[str]] = state_fips
        self.county_fips: Optional[List[str]] = county_fips
        self.output_path: Optional[str] = output_path
        
        # Set logging level for all loggers
        main_logger.setLevel(log_level)
        acs_logger.setLevel(log_level)
        tigris_logger.setLevel(log_level)
        places_logger.setLevel(log_level)
        integration_logger.setLevel(log_level)
        
        # Log initialization
        main_logger.info("Initializing IntegratedTractData class")
        main_logger.info(f"State FIPS filter: {state_fips}")
        main_logger.info(f"County FIPS filter: {county_fips}")
        main_logger.info(f"Output path: {output_path}")
        main_logger.info(f"ACS year: {ACS_YEAR}")
        
        # Initialize Census API client
        self.c: Census = Census(self.census_api_key)
        main_logger.info("Census API client initialized")
        
        # Setup dataframes to store results
        self.acs_data: Optional[pd.DataFrame] = None
        self.tigris_data: Optional[gpd.GeoDataFrame] = None
        self.places_data: Optional[pd.DataFrame] = None
        self.integrated_data: Optional[gpd.GeoDataFrame] = None
        main_logger.info("IntegratedTractData initialization complete")
    
    def fetch_acs5_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch the latest ACS5 data at the tract level.
        
        Returns:
        --------
        pandas.DataFrame or None
            Dataframe containing ACS5 data, or None if fetching failed
        """
        acs_logger.info(f"Starting ACS5 data fetch for year {ACS_YEAR}")
        print(f"Fetching ACS5 data for year {ACS_YEAR}...")
        
        start_time = time.time()
        
        # Define ACS5 variables to fetch
        # This is a representative list - modify as needed
        acs_variables: List[str] = [
            'B01001_001E',  # Total population
            'B01002_001E',  # Median age
            'B19013_001E',  # Median household income
            'B19083_001E',  # Gini Index of Income Inequality
            'B17001_002E',  # Population below poverty level
            'B25077_001E',  # Median home value
            'B25064_001E',  # Median gross rent
            'B15003_022E',  # Bachelor's degree
            'B15003_023E',  # Master's degree
            'B15003_024E',  # Professional degree
            'B15003_025E',  # Doctorate degree
            'B08301_001E',  # Total commuters
            'B08301_003E',  # Car commuters
            'B08301_010E',  # Public transit commuters
            'B08301_019E',  # Work from home
            'B25002_003E',  # Vacant housing units
            'B25002_001E',  # Total housing units
        ]
        
        acs_logger.info(f"Requesting {len(acs_variables)} ACS5 variables")
        acs_logger.debug(f"Variables: {acs_variables}")
        
        # Variable names for readability
        acs_variable_names: Dict[str, str] = {
            'B01001_001E': 'total_population',
            'B01002_001E': 'median_age',
            'B19013_001E': 'median_household_income',
            'B19083_001E': 'gini_index',
            'B17001_002E': 'poverty_population',
            'B25077_001E': 'median_home_value',
            'B25064_001E': 'median_rent',
            'B15003_022E': 'bachelors_degree',
            'B15003_023E': 'masters_degree',
            'B15003_024E': 'professional_degree',
            'B15003_025E': 'doctorate_degree',
            'B08301_001E': 'total_commuters',
            'B08301_003E': 'car_commuters',
            'B08301_010E': 'public_transit_commuters',
            'B08301_019E': 'work_from_home',
            'B25002_003E': 'vacant_housing_units',
            'B25002_001E': 'total_housing_units',
        }
        
        # Fetch data for all states or specified states
        all_tracts_data: List[Dict[str, Any]] = []
        
        if self.state_fips:
            state_list: List[str] = self.state_fips
            acs_logger.info(f"Fetching ACS5 data for specific states: {state_list}")
        else:
            # Get all states if not specified
            state_list: List[str] = [state.fips for state in states.STATES]
            acs_logger.info(f"Fetching ACS5 data for all {len(state_list)} states")
        
        for state_fips in state_list:
            state_start_time = time.time()
            try:
                acs_logger.info(f"Fetching ACS5 data for state FIPS {state_fips}...")
                print(f"Fetching ACS5 data for state FIPS {state_fips}...")
                
                # Get tract data for the state
                tracts_data = self.c.acs5.state_county_tract(
                    fields=acs_variables,
                    state_fips=state_fips,
                    county_fips="*",
                    tract="*",
                    year=ACS_YEAR
                )
                
                state_end_time = time.time()
                acs_logger.info(f"Retrieved {len(tracts_data)} tract records for state {state_fips} in {state_end_time - state_start_time:.2f} seconds")
                
                all_tracts_data.extend(tracts_data)
                
                # Respect API rate limits
                # time.sleep(1)
            
            except Exception as e:
                acs_logger.error(f"Error fetching ACS data for state {state_fips}: {e}")
                print(f"Error fetching ACS data for state {state_fips}: {e}")
        
        # Convert to DataFrame
        if all_tracts_data:
            acs_logger.info(f"Processing {len(all_tracts_data)} total tract records")
            
            df: pd.DataFrame = pd.DataFrame(all_tracts_data)
            acs_logger.info(f"Created DataFrame with shape {df.shape}")
            
            # Rename columns for readability
            df = df.rename(columns=acs_variable_names)
            acs_logger.info("Renamed columns for readability")
            
            # Create GEOID by concatenating state, county, and tract
            df['GEOID'] = df['state'] + df['county'] + df['tract']
            acs_logger.info("Created GEOID column for joining")
            
            # Convert numeric columns to proper types
            numeric_cols: List[str] = list(acs_variable_names.values())
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            acs_logger.info("Converted numeric columns to proper types")
            
            # Calculate additional metrics
            acs_logger.info("Calculating additional metrics")
            
            # Calculate additional metrics
            if 'bachelors_degree' in df.columns and 'masters_degree' in df.columns and \
               'professional_degree' in df.columns and 'doctorate_degree' in df.columns and \
               'total_population' in df.columns:
                df['higher_education_pct'] = (
                    df['bachelors_degree'] + df['masters_degree'] + 
                    df['professional_degree'] + df['doctorate_degree']
                ) / df['total_population'] * 100
                acs_logger.info("Added higher_education_pct column")
            
            if 'vacant_housing_units' in df.columns and 'total_housing_units' in df.columns:
                df['vacancy_rate'] = df['vacant_housing_units'] / df['total_housing_units'] * 100
                acs_logger.info("Added vacancy_rate column")
                
            if 'poverty_population' in df.columns and 'total_population' in df.columns:
                df['poverty_rate'] = df['poverty_population'] / df['total_population'] * 100
                acs_logger.info("Added poverty_rate column")
            
            # Check for missing values
            missing_values = df.isna().sum().sum()
            acs_logger.info(f"Total missing values in dataset: {missing_values}")
                
            self.acs_data = df
            end_time = time.time()
            acs_logger.info(f"Successfully fetched ACS5 data with {len(df)} tracts in {end_time - start_time:.2f} seconds")
            print(f"Successfully fetched ACS5 data with {len(df)} tracts.")
            
            # Save data if output path is provided
            if self.output_path:
                output_file: str = os.path.join(self.output_path, f"acs5_tract_data_{ACS_YEAR}.csv")
                df.to_csv(output_file, index=False)
                file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                acs_logger.info(f"Saved ACS5 data to {output_file} ({file_size_mb:.2f} MB)")
                print(f"Saved ACS5 data to {output_file}")
            
            end_time = time.time()
            acs_logger.info(f"ACS5 data fetch completed successfully in {end_time - start_time:.2f} seconds")
            return df
            
    def fetch_tigris_data(self) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch the latest TIGER/Line tract boundaries using pygris.
        
        Returns:
        --------
        geopandas.GeoDataFrame or None
            GeoDataFrame containing tract boundaries, or None if fetching failed
        """
        tigris_logger.info(f"Starting TIGER/Line data fetch for year {ACS_YEAR}")
        print("Fetching TIGER/Line tract boundaries...")
        
        start_time = time.time()
        
        try:
            # Create a cache directory if it doesn't exist
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
            os.makedirs(cache_dir, exist_ok=True)
            tigris_logger.info(f"Using cache directory: {cache_dir}")
            
            # Set the pygris cache directory globally
            # Note: pygris uses the PYGRIS_CACHE environment variable for cache location
            os.environ['PYGRIS_CACHE'] = cache_dir
            tigris_logger.info(f"Set PYGRIS_CACHE environment variable to: {cache_dir}")
            
            # Get tracts for the specified states or all states
            if self.state_fips:
                tigris_logger.info(f"Fetching TIGER/Line data for specific states: {self.state_fips}")
                all_tracts: gpd.GeoDataFrame = gpd.GeoDataFrame()
                
                for state_fips in self.state_fips:
                    state_start_time = time.time()
                    tigris_logger.info(f"Fetching TIGER/Line data for state FIPS {state_fips}...")
                    # Use pygris.tracts to get tract boundaries with caching enabled
                    # Note: pygris.tracts() doesn't accept cache_dir parameter directly
                    state_tracts: gpd.GeoDataFrame = pygris.tracts(
                        state=state_fips,
                        year=ACS_YEAR,
                        cb=True,  # Use cartographic boundaries (simplified)
                        cache=True   # Enable caching (uses PYGRIS_CACHE env var)
                    )
                    
                    state_end_time = time.time()
                    tigris_logger.info(f"Retrieved {len(state_tracts)} tract boundaries for state {state_fips} in {state_end_time - state_start_time:.2f} seconds")
                    
                    all_tracts = pd.concat([all_tracts, state_tracts])
            else:
                # Get tracts for all states with caching enabled
                tigris_logger.info(f"Fetching TIGER/Line data for all states")
                all_tracts_start_time = time.time()
                
                all_tracts: gpd.GeoDataFrame = pygris.tracts(
                    year=ACS_YEAR,
                    cb=True,  # Use cartographic boundaries (simplified)
                    cache=True   # Enable caching (uses PYGRIS_CACHE env var)
                )
                
                all_tracts_end_time = time.time()
                tigris_logger.info(f"Retrieved {len(all_tracts)} tract boundaries for all states in {all_tracts_end_time - all_tracts_start_time:.2f} seconds")
            
            # Process the retrieved data
            processing_start_time = time.time()
            tigris_logger.info(f"Processing {len(all_tracts)} tract boundaries")
            
            # Ensure GEOID is a string and properly formatted
            all_tracts['GEOID'] = all_tracts['GEOID'].astype(str)
            tigris_logger.info("Formatted GEOID column as string")
            
            # Select relevant columns
            selected_cols: List[str] = ['GEOID', 'NAME', 'NAMELSAD', 'STATEFP', 'COUNTYFP', 'TRACTCE', 'ALAND', 'AWATER', 'geometry']
            all_tracts = all_tracts[selected_cols]
            tigris_logger.info(f"Selected {len(selected_cols)} relevant columns")
            
            # Calculate area in square kilometers
            area_calc_start = time.time()
            tigris_logger.info("Calculating area in square kilometers...")
            all_tracts['area_sq_km'] = all_tracts.to_crs(epsg=3857).area / 10**6
            area_calc_end = time.time()
            tigris_logger.info(f"Area calculation completed in {area_calc_end - area_calc_start:.2f} seconds")
            
            processing_end_time = time.time()
            tigris_logger.info(f"Processing completed in {processing_end_time - processing_start_time:.2f} seconds")
            
            self.tigris_data = all_tracts
            end_time = time.time()
            tigris_logger.info(f"Successfully fetched TIGER/Line data with {len(all_tracts)} tracts in {end_time - start_time:.2f} seconds")
            print(f"Successfully fetched TIGER/Line data with {len(all_tracts)} tracts.")
            
            # Save data if output path is provided
            if self.output_path:
                save_start_time = time.time()
                output_file: str = os.path.join(self.output_path, f"tigris_tract_boundaries_{ACS_YEAR}.geojson")
                tigris_logger.info(f"Saving TIGER/Line data to {output_file}")
                
                all_tracts.to_file(output_file, driver='GeoJSON')
                
                save_end_time = time.time()
                file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                tigris_logger.info(f"Saved TIGER/Line data to {output_file} ({file_size_mb:.2f} MB) in {save_end_time - save_start_time:.2f} seconds")
                print(f"Saved TIGER/Line data to {output_file}")
                
            return all_tracts
        
        except Exception as e:
            end_time = time.time()
            tigris_logger.error(f"Error fetching TIGER/Line data: {e}")
            tigris_logger.error(f"TIGER/Line data fetch failed after {end_time - start_time:.2f} seconds")
            print(f"Error fetching TIGER/Line data: {e}")
            return None

    @staticmethod
    def available_state_fips_codes() -> Dict[str, int]:
        """
        Returns a dictionary mapping state FIPS codes to state abbreviations from the ACS data file.
        
        Returns:
        --------
        Dict[str, int]
            Dictionary mapping state FIPS codes to state abbreviations (e.g., {'01': 'AL', '06': 'CA'})
        """
        acs_logger.info("Starting retrieval of state FIPS codes from ACS data")
        print("Retrieving state FIPS codes from ACS data...")
        
        start_time = time.time()
        
        try:
            # Construct the path to the ACS data file
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            acs_file_path = os.path.join(output_dir, f"acs5_tract_data_{ACS_YEAR}.csv")
            acs_logger.info(f"Looking for ACS data file at {acs_file_path}")
            
            # Check if the file exists
            if not os.path.exists(acs_file_path):
                acs_logger.error(f"ACS data file not found at {acs_file_path}")
                print(f"ACS data file not found at {acs_file_path}")
                return {}
            
            # Read the CSV file
            acs_logger.info(f"Reading ACS data file from {acs_file_path}")
            file_size_mb = os.path.getsize(acs_file_path) / (1024 * 1024)
            acs_logger.info(f"ACS data file size: {file_size_mb:.2f} MB")
            
            df = pd.read_csv(acs_file_path)
            acs_logger.info(f"Loaded ACS data with shape {df.shape}")
            
            # Extract unique state FIPS codes
            if 'state' in df.columns:
                state_fips_codes = sorted(df['state'].unique().tolist())
                acs_logger.info(f"Found {len(state_fips_codes)} unique state FIPS codes")
                
                # Create a dictionary mapping FIPS codes to state abbreviations
                acs_logger.info("Converting state FIPS codes to abbreviations")
                fips_to_abbr = {}
                conversion_start_time = time.time()
                
                for fips in state_fips_codes:
                    try:
                        # Convert FIPS code to properly formatted string (2-digit with leading zero if needed)
                        fips_str = str(fips).zfill(2)
                        acs_logger.debug(f"Processing FIPS code: {fips} (formatted as {fips_str})")
                        
                        # Find the state by FIPS code and get its abbreviation
                        state = states.lookup(fips_str)
                        if state:
                            fips_to_abbr[fips] = state.abbr
                            acs_logger.debug(f"Converted FIPS {fips_str} to {state.abbr}")
                        else:
                            acs_logger.warning(f"Could not find state with FIPS code: {fips_str}")
                            print(f"Could not find state with FIPS code: {fips_str}")
                    except Exception as e:
                        acs_logger.error(f"Error converting FIPS code {fips} to abbreviation: {e}")
                        print(f"Error converting FIPS code {fips} to abbreviation: {e}")
                
                conversion_end_time = time.time()
                acs_logger.info(f"FIPS code conversion completed in {conversion_end_time - conversion_start_time:.2f} seconds")
                acs_logger.info(f"Successfully mapped {len(fips_to_abbr)} state FIPS codes to abbreviations")
                
                end_time = time.time()
                acs_logger.info(f"State FIPS codes retrieval completed in {end_time - start_time:.2f} seconds")
                print(f"Successfully retrieved {len(fips_to_abbr)} state FIPS codes.")
                
                return fips_to_abbr
            else:
                acs_logger.error("'state' column not found in ACS data file")
                print("'state' column not found in ACS data file")
                return {}
        except Exception as e:
            end_time = time.time()
            acs_logger.error(f"Error retrieving state FIPS codes: {e}")
            acs_logger.error(f"State FIPS codes retrieval failed after {end_time - start_time:.2f} seconds")
            print(f"Error retrieving state FIPS codes: {e}")
            return {}
            
    def fetch_places_data(self) -> Optional[pd.DataFrame]:
        """
        Fetch CDC PLACES health data at the census tract level.
        
        Returns:
        --------
        pandas.DataFrame or None
            DataFrame containing CDC PLACES health data, or None if fetching failed
        """
        places_logger.info("Starting CDC PLACES data fetch")
        print("Fetching CDC PLACES health data...")
        
        start_time = time.time()
        all_states_data = []
        processed_states = 0
        
        try:
            # Get the state FIPS to abbreviation mapping
            state_fips_mapping = self.available_state_fips_codes()
            places_logger.info(f"Found {len(state_fips_mapping)} states to process")
            
            # Check if we should filter to specific states based on self.state_fips
            if self.state_fips:
                places_logger.info(f"Filtering to only process states in self.state_fips: {self.state_fips}")
                # Filter the mapping to only include states in self.state_fips
                filtered_mapping = {fips: abbr for fips, abbr in state_fips_mapping.items() if fips in self.state_fips}
                if filtered_mapping:
                    state_fips_mapping = filtered_mapping
                    places_logger.info(f"Filtered to {len(state_fips_mapping)} states")
                else:
                    places_logger.warning(f"None of the specified state_fips {self.state_fips} were found in available states")
            
            # Process each state individually
            for state_fips_2, state_abbr in state_fips_mapping.items():
                places_logger.info(f"Processing state: {state_abbr} (FIPS: {state_fips_2})")
                
                # CDC PLACES API endpoint for census tract data
                places_logger.info(f"Setting up CDC PLACES API request for state: {state_abbr}")
                base_url = "https://data.cdc.gov/resource/cwsq-ngmh.json"
                
                # Parameters for the API request - no API key needed
                params = {
                    "$limit": 10_000_000,
                    "$order": "locationname",
                    "stateabbr": state_abbr,
                    # Using full fields instead of select to get complete data structure
                }
                
                # No headers needed since API key is not required
                headers = {}
                places_logger.info("CDC PLACES API does not require an API key")
                
                # Make the API request with headers and params
                places_logger.info(f"Making CDC PLACES API request to {base_url} for state {state_abbr}")
                request_start_time = time.time()
                response = requests.get(base_url, params=params, headers=headers)
                request_end_time = time.time()
                places_logger.info(f"API request for state {state_abbr} completed in {request_end_time - request_start_time:.2f} seconds")
                
                # Check if request was successful
                if response.status_code == 200:
                    places_logger.info(f"CDC PLACES API request for state {state_abbr} successful")
                    data = response.json()
                    places_logger.info(f"Retrieved {len(data)} records from CDC PLACES API for state {state_abbr}")
                    
                    # Skip processing if no data was returned
                    if not data:
                        places_logger.warning(f"No data returned for state {state_abbr}, skipping")
                        continue
                    
                    # Process the data
                    processing_start_time = time.time()
                    places_logger.info(f"Processing CDC PLACES data for state {state_abbr}")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(data)
                    places_logger.info(f"Created DataFrame for state {state_abbr} with {len(df)} rows and {len(df.columns)} columns")
                    
                    # Add to our collection of state data
                    all_states_data.append(df)
                    processed_states += 1
                    places_logger.info(f"Added state {state_abbr} data to collection (processed {processed_states} of {len(state_fips_mapping)} states)")
                    
                else:
                    places_logger.error(f"CDC PLACES API request for state {state_abbr} failed with status code: {response.status_code}")
                    places_logger.error(f"Response: {response.text}")
                    print(f"Error: CDC PLACES API request for state {state_abbr} failed with status code {response.status_code}")
                    # Continue with other states even if one fails
                    continue
            
            # Combine all state data
            if not all_states_data:
                places_logger.error("No data was successfully retrieved for any state")
                print("Error: No CDC PLACES data could be retrieved")
                return None
            
            places_logger.info(f"Combining data from {len(all_states_data)} states")
            combined_df = pd.concat(all_states_data, ignore_index=True)
            places_logger.info(f"Combined data has {len(combined_df)} rows")
            
            # Process the combined data
            processing_start_time = time.time()
            places_logger.info("Processing combined CDC PLACES data")
            
            # Ensure data_value is numeric
            combined_df['data_value'] = pd.to_numeric(combined_df['data_value'], errors='coerce')
            
            # Pivot the data to have measures as columns
            places_logger.info("Pivoting combined data to have measures as columns")
            pivot_df = combined_df.pivot_table(
                index='locationid',
                columns='measureid',  # Using measureid instead of measure for more consistent column names
                values='data_value',
                aggfunc='first'
            ).reset_index()
            
            # Add some additional columns from the original data that might be useful
            # Group by locationid and get the first occurrence of these columns
            additional_cols = ['stateabbr', 'countyname', 'countyfips']
            if all(col in combined_df.columns for col in additional_cols):
                places_logger.info("Adding additional geographic columns")
                geo_info = combined_df[['locationid'] + additional_cols].drop_duplicates('locationid').set_index('locationid')
                pivot_df = pivot_df.merge(geo_info, on='locationid', how='left')
            
            # Rename locationid to GEOID for consistency with other datasets
            pivot_df = pivot_df.rename(columns={'locationid': 'GEOID'})
            places_logger.info("Renamed 'locationid' to 'GEOID' for consistency")
            
            # Ensure GEOID is a string
            pivot_df['GEOID'] = pivot_df['GEOID'].astype(str)
            places_logger.info("Formatted GEOID column as string")
            
            # Add prefix to health measure columns to avoid conflicts
            measure_cols = [col for col in pivot_df.columns if col not in ['GEOID'] + additional_cols]
            rename_dict = {col: f'health_{col.lower()}' for col in measure_cols}
            pivot_df = pivot_df.rename(columns=rename_dict)
            places_logger.info(f"Added 'health_' prefix to {len(measure_cols)} measure columns")
            
            processing_end_time = time.time()
            places_logger.info(f"Processing completed in {processing_end_time - processing_start_time:.2f} seconds")
            
            # Store the processed data
            self.places_data = pivot_df
            end_time = time.time()
            places_logger.info(f"Successfully fetched CDC PLACES data with {len(pivot_df)} tracts from {processed_states} states in {end_time - start_time:.2f} seconds")
            print(f"Successfully fetched CDC PLACES data with {len(pivot_df)} tracts from {processed_states} states.")
            
            # Save data if output path is provided
            if self.output_path:
                save_start_time = time.time()
                output_file: str = os.path.join(self.output_path, "cdc_places_data.csv")
                places_logger.info(f"Saving CDC PLACES data to {output_file}")
                
                pivot_df.to_csv(output_file, index=False)
                
                save_end_time = time.time()
                file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                places_logger.info(f"Saved CDC PLACES data to {output_file} ({file_size_mb:.2f} MB) in {save_end_time - save_start_time:.2f} seconds")
                print(f"Saved CDC PLACES data to {output_file}")
            
            return pivot_df
                
        except Exception as e:
            end_time = time.time()
            places_logger.error(f"Error fetching CDC PLACES data: {e}")
            places_logger.error(f"CDC PLACES data fetch failed after {end_time - start_time:.2f} seconds")
            print(f"Error fetching CDC PLACES data: {e}")
            return None
        
    def integrate_data(self) -> Optional[gpd.GeoDataFrame]:
        """
        Integrate all data sources using GEOID as the common identifier.
        
        Returns:
        --------
        geopandas.GeoDataFrame or None
            Integrated dataset with all sources joined, or None if integration failed
        """
        integration_logger.info("Starting data integration process")
        print("Integrating datasets...")
        
        start_time = time.time()
        
        try:
            # Ensure all required datasets are available
            integration_logger.info("Checking availability of required datasets")
            
            # First, check ACS5 data
            if self.acs_data is None:
                integration_logger.info("ACS5 data not fetched yet, fetching now")
                print("ACS5 data not fetched yet. Fetching now...")
                self.fetch_acs5_data()
                if self.acs_data is not None:
                    integration_logger.info("Successfully fetched ACS5 data")
                else:
                    integration_logger.warning("Failed to fetch ACS5 data")
            
            # Next, check TIGER/Line data
            if self.tigris_data is None:
                integration_logger.info("TIGER/Line data not fetched yet, fetching now")
                print("TIGER/Line data not fetched yet. Fetching now...")
                self.fetch_tigris_data()
                if self.tigris_data is not None:
                    integration_logger.info("Successfully fetched TIGER/Line data")
                else:
                    integration_logger.warning("Failed to fetch TIGER/Line data")
            
            # Last, check CDC PLACES data
            if self.places_data is None:
                integration_logger.info("CDC PLACES data not fetched yet, fetching now")
                print("CDC PLACES data not fetched yet. Fetching now...")
                self.fetch_places_data()
                if self.places_data is not None:
                    integration_logger.info("Successfully fetched CDC PLACES data")
                else:
                    integration_logger.warning("Failed to fetch CDC PLACES data")
            
            # Check which datasets we have successfully fetched
            available_datasets: List[str] = []
            missing_datasets: List[str] = []
            
            if self.acs_data is not None:
                available_datasets.append("ACS5")
            else:
                missing_datasets.append("ACS5")
                
            if self.tigris_data is not None:
                available_datasets.append("TIGER/Line")
            else:
                missing_datasets.append("TIGER/Line")
                
            if self.places_data is not None:
                available_datasets.append("PLACES")
            else:
                missing_datasets.append("PLACES")
                
            integration_logger.info(f"Available datasets: {available_datasets}")
            if missing_datasets:
                integration_logger.warning(f"Missing datasets: {missing_datasets}")
            
            # We need at least ACS5 and TIGER/Line for basic integration
            required_datasets = ["ACS5", "TIGER/Line"]
            missing_required = [ds for ds in required_datasets if ds not in available_datasets]
            
            if missing_required:
                integration_logger.error(f"Cannot integrate data due to missing required datasets: {missing_required}")
                print(f"Warning: Could not integrate data due to missing datasets: {missing_required}")
                return None
                
            integration_logger.info("All required datasets available, proceeding with integration")
            
            # Start with TIGER/Line data as the base (it has geometry)
            integration_logger.info("Using TIGER/Line data as the base for integration")
            base_start_time = time.time()
            integrated_gdf: gpd.GeoDataFrame = self.tigris_data.copy()
            integration_logger.info(f"Copied TIGER/Line data as base ({len(integrated_gdf)} tracts)")
            print(f"Using TIGER/Line data as base ({len(integrated_gdf)} tracts)")
            
            # Join ACS5 data
            if "ACS5" in available_datasets:
                integration_logger.info(f"Joining ACS5 data ({len(self.acs_data)} tracts) to base")
                print(f"Joining ACS5 data ({len(self.acs_data)} tracts)")
                acs_join_start = time.time()
                integrated_gdf = integrated_gdf.merge(self.acs_data, on='GEOID', how='left')
                acs_join_end = time.time()
                integration_logger.info(f"ACS5 data joined in {acs_join_end - acs_join_start:.2f} seconds")
            
            # Join PLACES data if available
            if "PLACES" in available_datasets:
                integration_logger.info(f"Joining CDC PLACES data ({len(self.places_data)} tracts) to base")
                print(f"Joining CDC PLACES data ({len(self.places_data)} tracts)")
                
                places_join_start = time.time()
                # Rename columns to avoid conflicts
                integration_logger.info("Renaming PLACES columns to avoid conflicts")
                places_data_renamed = self.places_data.rename(columns={
                    col: f"{col}_places" for col in self.places_data.columns 
                    if col not in ['GEOID']
                })
                
                integrated_gdf = integrated_gdf.merge(places_data_renamed, on='GEOID', how='left')
                places_join_end = time.time()
                integration_logger.info(f"CDC PLACES data joined in {places_join_end - places_join_start:.2f} seconds")
            
            # Check for missing data after joining
            integration_logger.info(f"Integrated data has {len(integrated_gdf)} tracts with {len(integrated_gdf.columns)} columns")
            print(f"Integrated data has {len(integrated_gdf)} tracts with {len(integrated_gdf.columns)} columns")
            
            # Analyze missing values
            missing_analysis_start = time.time()
            integration_logger.info("Analyzing missing values in integrated data")
            
            # Count missing values by source
            if "ACS5" in available_datasets and "PLACES" in available_datasets:
                acs_cols: List[str] = [col for col in self.acs_data.columns if col != 'GEOID']
                places_cols: List[str] = [col for col in places_data_renamed.columns if col != 'GEOID']
                
                missing_acs: int = integrated_gdf[acs_cols].isna().sum().sum()
                missing_places: int = integrated_gdf[places_cols].isna().sum().sum()
                
                missing_analysis_end = time.time()
                integration_logger.info(f"Missing value analysis completed in {missing_analysis_end - missing_analysis_start:.2f} seconds")
                integration_logger.info(f"Missing ACS values: {missing_acs}, Missing PLACES values: {missing_places}")
                
                print(f"Integration complete. Dataset has {len(integrated_gdf)} tracts.")
                print(f"Missing ACS values: {missing_acs}, Missing PLACES values: {missing_places}")
            
            self.integrated_data = integrated_gdf
            integration_logger.info("Assigned integrated data to class attribute")
            
            # Save integrated data if output path is provided
            if self.output_path:
                integration_logger.info("Saving integrated data to output files")
                save_start_time = time.time()
                
                # Save as GeoPackage to preserve geometry
                output_file_gpkg: str = os.path.join(self.output_path, f"integrated_tract_data_{ACS_YEAR}.gpkg")
                integration_logger.info(f"Saving GeoPackage to {output_file_gpkg}")
                integrated_gdf.to_file(output_file_gpkg, driver='GPKG')
                gpkg_size_mb = os.path.getsize(output_file_gpkg) / (1024 * 1024)
                integration_logger.info(f"Saved GeoPackage file ({gpkg_size_mb:.2f} MB)")
                
                # Also save as CSV for non-spatial use
                output_file_csv: str = os.path.join(self.output_path, f"integrated_tract_data_{ACS_YEAR}.csv")
                integration_logger.info(f"Saving CSV to {output_file_csv}")
                # Drop geometry for CSV version
                integrated_gdf_no_geom: pd.DataFrame = integrated_gdf.drop(columns=['geometry'])
                integrated_gdf_no_geom.to_csv(output_file_csv, index=False)
                csv_size_mb = os.path.getsize(output_file_csv) / (1024 * 1024)
                integration_logger.info(f"Saved CSV file ({csv_size_mb:.2f} MB)")
                
                save_end_time = time.time()
                integration_logger.info(f"Saved all output files in {save_end_time - save_start_time:.2f} seconds")
                
                print(f"Saved integrated data to:")
                print(f"  - {output_file_gpkg} (with geometry)")
                print(f"  - {output_file_csv} (without geometry)")
            
            end_time = time.time()
            integration_logger.info(f"Data integration completed successfully in {end_time - start_time:.2f} seconds")
            return integrated_gdf
            
        except Exception as e:
            end_time = time.time()
            integration_logger.error(f"Error integrating datasets: {e}")
            integration_logger.error(f"Integration failed after {end_time - start_time:.2f} seconds")
            print(f"Error integrating datasets: {e}")
            return None

def main() -> None:
    """
    Main function to demonstrate the usage of the IntegratedTractData class.
    """
    # Define output directory
    output_dir: str = r"C:\Users\jacob\DSC680 - Applied Data Science\Project #2 (Weeks 5 thru 8)\Week #5\output"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the data integration class
    # Example 1: Get data for all states
    tract_data: IntegratedTractData = IntegratedTractData(
        census_api_key=CENSUS_API_KEY,
        output_path=output_dir
    )
    
    # Example 2: Get data for specific states
    # Uncomment and modify as needed
    # tract_data = IntegratedTractData(
    #     census_api_key=CENSUS_API_KEY,
    #     state_fips=['06', '36'],  # California and New York
    #     output_path=output_dir
    # )
    
    # Fetch and integrate all data
    print("Starting data integration process...")
    
    # Option 1: Fetch all data separately then integrate
    # acs_data: Optional[pd.DataFrame] = tract_data.fetch_acs5_data()
    # tigris_data: Optional[gpd.GeoDataFrame] = tract_data.fetch_tigris_data()
    # places_data: Optional[pd.DataFrame] = tract_data.fetch_places_data()
    # integrated_data: Optional[gpd.GeoDataFrame] = tract_data.integrate_data()
    
    # Option 2: Directly integrate (will fetch data as needed)
    integrated_data: Optional[gpd.GeoDataFrame] = tract_data.integrate_data()
    
    if integrated_data is not None:
        print(f"Successfully created integrated dataset with {len(integrated_data)} tracts.")
        
        # Display summary statistics
        summary_columns: List[str] = [
            'total_population', 'median_age', 'median_household_income', 
            'poverty_rate', 'median_home_value', 'higher_education_pct'
        ]
        
        # Only include columns that exist in the integrated data
        available_summary_cols: List[str] = [col for col in summary_columns if col in integrated_data.columns]
        
        if available_summary_cols:
            print("\nSummary Statistics:")
            print(integrated_data[available_summary_cols].describe())
    else:
        print("Failed to create integrated dataset.")



if __name__ == "__main__":
    main()