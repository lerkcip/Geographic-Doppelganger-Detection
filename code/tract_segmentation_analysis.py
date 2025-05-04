"""
Census Tract Segmentation Analysis

This script analyzes the integrated tract data to find patterns and groupings
based on demographic and socioeconomic characteristics, without using
geographical information during clustering. After clustering, geographic
information is reintroduced to identify "doppelganger" tracts (similar
characteristics in different locations).

The script implements:
    1. Advanced data loading and preprocessing
    2. Statistical analysis and feature selection
    3. Dimensionality reduction (PCA, UMAP)
    4. Cluster analysis (KMeans, GaussianMixture, HDBSCAN)
    5. XGBoost classification with hyperparameter optimization
    6. Feature importance analysis
    7. Geographic doppelganger identification (sampling, NearestNeighbors)

Author: Jacob
Date: April 26, 2025
Version: 3.7.1
"""

# Standard library imports
import os
import warnings
import platform
import subprocess
from time import time
from typing import Dict, List, Optional, Tuple, Union, Any, Set, cast
import sys

# Third-party imports
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# Progress bar imports
try:
    from tqdm import tqdm
    from tqdm.notebook import tqdm as tqdm_notebook
    TQDM_AVAILABLE = True
except ImportError:
    # Define fallback if tqdm is not available
    print("tqdm not available, installing...")
    try:
        import pip
        pip.main(['install', 'tqdm'])
        from tqdm import tqdm
        from tqdm.notebook import tqdm as tqdm_notebook
        TQDM_AVAILABLE = True
    except:
        print("Error installing tqdm. Progress bars will not be available.")
        # Define a simple fallback tqdm function
        def tqdm(iterable, **kwargs):
            return iterable
        tqdm_notebook = tqdm
        TQDM_AVAILABLE = False

# Machine learning imports
# - Preprocessing
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer

# - Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
try:
    import hdbscan  # type: ignore
except ImportError:
    hdbscan = None
try:
    import umap  # type: ignore
except ImportError:
    umap = None

# - Clustering
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors

# - Classification and feature selection
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline

# - Metrics and model selection
from sklearn.metrics import (
    silhouette_score, 
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    f1_score
)
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold

# - Outlier detection
from sklearn.covariance import EllipticEnvelope

# - XGBoost
import xgboost as xgb
from xgboost import XGBClassifier

# Configuration and constants
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set up paths
# Detect if running in Colab
IN_COLAB = 'google.colab' in sys.modules or os.environ.get('IN_COLAB') == '1' or '--colab' in sys.argv

if IN_COLAB:
    # Set your base directory as you mounted it in Google Drive
    BASE_DIR = "/content/drive/Othercomputers/My Laptop/Project #2 (Weeks 5 thru 8)/Week #5"
else:
    # Local Windows path
    BASE_DIR = r"C:\Users\jacob\DSC680 - Applied Data Science\Project #2 (Weeks 5 thru 8)\Week #5"

INPUT_PATH = os.path.join(BASE_DIR, "output")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "segmentation")
INPUT_FILE: str = 'integrated_tract_data_2022.gpkg'

# GPU/CUDA Configuration
GPU_ENABLED: bool = False  # Will be set dynamically
CUDA_AVAILABLE: bool = False  # Will be set dynamically

# Progress bar configuration
USE_PROGRESS_BARS: bool = True  # Set to False to disable all progress bars
IS_NOTEBOOK: bool = 'ipykernel' in sys.modules  # Auto-detect if running in a notebook

# Analysis parameters
MISSING_THRESHOLD: float = 0.8  # Drop columns with more than 80% missing values
LOW_VAR_THRESHOLD: float = 0.01  # Threshold for low variance columns
EXTREME_CORR_THRESHOLD: float = 0.95  # Threshold for extreme correlations
OUTLIER_IQR_FACTOR: float = 1.5  # Factor for IQR-based outlier detection
EXTREME_OUTLIER_THRESHOLD: float = -0.5  # Threshold for extreme outliers in Isolation Forest
MIN_CLUSTERS: int = 2  # Minimum number of clusters
MAX_CLUSTERS: int = 15  # Maximum number of clusters
# Default cluster counts
CLUSTER_COUNTS: List[int] = [10, 11, 12]  # Default cluster counts to evaluate
DEFAULT_CLUSTER_COUNT: int = CLUSTER_COUNTS[0]  # Primary cluster count

# Doppelganger evaluation parameters
DOPPELGANGER_SAMPLE_SIZE: int = 10  # Number of doppelganger pairs per state
DOPPELGANGER_SIMILARITY_THRESHOLD: float = 0.0  # Minimum similarity to include
SIMILARITY_METRIC: str = "cosine"  # Metric for similarity calculation
FEATURE_WEIGHT_DEFAULT: float = 1.0  # Default weight for all features
FEATURE_WEIGHTS: Dict[str, float] = {}  # Feature-specific weights; defaults to FEATURE_WEIGHT_DEFAULT

## Algorithm options (comment out to disable)
CLUSTER_ALGOS: List[str] = ["kmeans", "gmm", "hdbscan"]  # Clustering methods
DR_METHODS: List[str] = ["pca", "umap"]  # Dimensionality reduction methods
DOPPELGANGER_METHODS: List[str] = ["sampling", "nearest_neighbors"]  # Doppelganger search methods

# Check for GPU/CUDA availability
def check_gpu_availability() -> Tuple[bool, bool]:
    """Check if GPU and CUDA are available for computation acceleration.
    
    Returns:
        Tuple[bool, bool]: (gpu_enabled, cuda_available)
    """
    gpu_enabled = False
    cuda_available = False
    
    try:
        # Check for NVIDIA GPU using system-specific methods
        if platform.system() == "Windows":
            # On Windows, use nvidia-smi
            try:
                nvidia_smi_output = subprocess.check_output(["nvidia-smi"], 
                                     stderr=subprocess.DEVNULL,
                                     universal_newlines=True)
                gpu_enabled = True
                if "CUDA Version" in nvidia_smi_output:
                    cuda_available = True
            except (subprocess.SubprocessError, FileNotFoundError):
                pass
        
        # Also try checking within Python using available packages
        try:
            import torch
            if torch.cuda.is_available():
                gpu_enabled = True
                cuda_available = True
        except ImportError:
            pass
            
        try:
            import tensorflow as tf
            if tf.test.is_gpu_available(cuda_only=True):
                gpu_enabled = True
                cuda_available = True
        except (ImportError, AttributeError):
            # TF 2.x uses different API
            try:
                import tensorflow as tf
                if len(tf.config.list_physical_devices('GPU')) > 0:
                    gpu_enabled = True
                    cuda_available = True
            except (ImportError, AttributeError):
                pass
                
        # Check via XGBoost itself
        try:
            import xgboost as xgb
            from xgboost import XGBClassifier
            # Try creating a small model with GPU parameters
            try:
                temp_model = XGBClassifier(tree_method='gpu_hist', n_estimators=1)
                temp_model.fit(np.array([[0, 0], [1, 1]]), np.array([0, 1]))
                gpu_enabled = True
                cuda_available = True
            except Exception:
                pass
        except ImportError:
            pass
    
    except Exception as e:
        print(f"Warning: Error checking GPU availability: {e}")
    
    return gpu_enabled, cuda_available

# Set GPU flags
GPU_ENABLED, CUDA_AVAILABLE = check_gpu_availability()

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

print(f"GPU Enabled: {GPU_ENABLED}")
print(f"CUDA Available: {CUDA_AVAILABLE}")

class TractSegmentationAnalysis:
    """Class for analyzing and segmenting census tract data.
    
    This class provides methods for loading, preprocessing, and analyzing census tract data
    to identify meaningful clusters of similar areas based on socioeconomic and demographic
    characteristics. It employs advanced techniques for data cleaning, feature selection,
    dimensionality reduction, and machine learning to generate high-quality segmentation.
    
    Attributes:
        input_path (str): Directory path where input data is located
        input_file (str): Filename of the input geopackage file
        output_path (str): Directory path where analysis outputs will be saved
        data (pd.DataFrame): Original loaded data (without preprocessing)
        geo_data (gpd.GeoDataFrame): Original data with geometry information
        numeric_data (pd.DataFrame): Processed data for analysis (numeric features only)
        id_data (pd.DataFrame): Identifier columns from the original dataset
        cluster_results (pd.DataFrame): Original data with cluster assignments
        pca_results (pd.DataFrame): Results of PCA dimensionality reduction
        feature_importance (pd.DataFrame): Feature importance rankings
        best_model (XGBClassifier): Best XGBoost classifier from hyperparameter tuning
        best_params (dict): Best hyperparameters from grid search
        cluster_labels (np.ndarray): Cluster assignments for each observation
        evaluation_metrics (dict): Performance metrics for the classifier
    """
    
    def __init__(self, input_path: str = INPUT_PATH, 
             input_file: str = INPUT_FILE, 
             output_path: str = OUTPUT_PATH,
             use_gpu: bool = GPU_ENABLED) -> None:
        """Initialize the TractSegmentationAnalysis class.
        
        Args:
            input_path: Directory path containing the input data file
            input_file: Name of the geopackage file to analyze
            output_path: Directory path where output files will be saved
            use_gpu: Whether to use GPU acceleration if available
        """
        self.input_path: str = input_path
        self.input_file: str = input_file
        self.output_path: str = output_path
        self.use_gpu: bool = use_gpu and GPU_ENABLED
        self.cuda_available: bool = CUDA_AVAILABLE
        
        # Log GPU status
        if self.use_gpu and self.cuda_available:
            print("CUDA GPU acceleration enabled")
        elif self.use_gpu and not self.cuda_available:
            print("GPU acceleration requested but CUDA not available - using CPU")
            self.use_gpu = False
        else:
            print("Using CPU for computations")
        
        # Data containers
        self.data: Optional[pd.DataFrame] = None
        self.geo_data: Optional[gpd.GeoDataFrame] = None  # Original data with geometry
        self.numeric_data: Optional[pd.DataFrame] = None  # Processed data for clustering
        self.id_data: Optional[pd.DataFrame] = None
        
        # Analysis results
        self.cluster_results: Optional[pd.DataFrame] = None
        self.pca_results: Optional[pd.DataFrame] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        
        # Model artifacts
        self.best_model: Optional[XGBClassifier] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.cluster_labels: Optional[np.ndarray] = None
        self.evaluation_metrics: Dict[str, Any] = {}
        
    def load_data(self) -> bool:
        """Load the geopackage data and perform initial data inspection.
        
        This method loads the geospatial data from the specified geopackage file,
        performs basic data inspection, and saves column names for reference.
        
        Returns:
            bool: True if data was loaded successfully, False otherwise
        """
        file_path: str = os.path.join(self.input_path, self.input_file)
        print(f"Loading data from {file_path}...")
        
        try:
            # Load the geospatial data
            self.geo_data = gpd.read_file(file_path)
            
            if self.geo_data is None or self.geo_data.empty:
                print("Error: Loaded GeoDataFrame is empty")
                return False
                
            print(f"Successfully loaded data with shape: {self.geo_data.shape}")
            
            # Create a copy without geometry for analysis
            self.data = self.geo_data.copy()
            
            # Basic info about the dataset
            print("\n--- Dataset Overview ---")
            
            tract_count: int = len(self.data)
            attribute_count: int = len(self.data.columns)
            
            print(f"Number of census tracts: {tract_count}")
            
            print(f"Number of attributes: {attribute_count}")
            
            # Save the column names to a text file for reference
            column_file_path: str = os.path.join(self.output_path, "column_names.txt")
            
            with open(column_file_path, "w") as f:
                for col in self.data.columns:
                    f.write(f"{col}\n")
            
            print(f"Column names saved to: {column_file_path}")
            return True
            
        except FileNotFoundError:
            print(f"Error: File not found - {file_path}")
            
            return False
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            
            return False
    
    def preprocess_data(self):
        """Prepare data for analysis with advanced data quality techniques:
        1. Handling missing values with sophisticated imputation methods
        2. Detecting and handling outliers
        3. Ensuring values fall within reasonable bounds
        4. Comprehensive data quality checks
        """
        if self.data is None:
            print("No data loaded. Please load data first.")
            return False
        
        print("\n--- Preprocessing Data ---")
        
        # 1. Extract identifier columns before preprocessing
        id_columns = ['GEOID', 'NAME', 'NAMELSAD', 'STATEFP', 'COUNTYFP', 'TRACTCE']
        geo_column = 'geometry'
        
        # Save original identifiers for later use
        self.id_data = self.data[id_columns].copy() if all(col in self.data.columns for col in id_columns) else None
        
        # 2. Determine which columns are numeric
        numeric_columns = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Remove id columns from numeric columns if they exist there
        numeric_columns = [col for col in numeric_columns if col not in id_columns and col != geo_column]
        
        print(f"Number of numeric columns for analysis: {len(numeric_columns)}")
        
        # 3. Create a dataset with only numeric columns
        numeric_data = self.data[numeric_columns].copy()
        
        # 4. Check for missing values - detailed analysis
        missing_data = numeric_data.isnull().sum()
        missing_cols = missing_data[missing_data > 0]
        
        if not missing_cols.empty:
            print(f"Found {len(missing_cols)} columns with missing values")
            print("Missing value statistics:")
            print(f"  Min missing: {missing_cols.min()} ({missing_cols.min()/len(numeric_data)*100:.2f}%)")
            print(f"  Max missing: {missing_cols.max()} ({missing_cols.max()/len(numeric_data)*100:.2f}%)")
            print(f"  Average missing: {missing_cols.mean():.1f} ({missing_cols.mean()/len(numeric_data)*100:.2f}%)")
            
            # Save missing data analysis to CSV
            missing_data_analysis = pd.DataFrame({
                'Column': missing_cols.index,
                'Missing_Count': missing_cols.values,
                'Missing_Percent': (missing_cols.values / len(numeric_data) * 100)
            })
            missing_data_analysis.to_csv(os.path.join(self.output_path, 'missing_data_analysis.csv'), index=False)
            
            # Drop columns with too many missing values (>80%)
            high_missing_cols = missing_cols[missing_cols/len(numeric_data) > 0.8].index.tolist()
            if high_missing_cols:
                print(f"Dropping {len(high_missing_cols)} columns with >80% missing values")
                numeric_data = numeric_data.drop(columns=high_missing_cols)
        else:
            print("No missing values found in the dataset")
        
        # 5. Remove constant and near-constant columns (low variance)
        # Calculate variance for each column
        variances = numeric_data.var()
        low_var_threshold = 0.01  # Threshold for what constitutes "low variance"
        low_var_cols = variances[variances < low_var_threshold].index.tolist()
        
        if low_var_cols:
            print(f"Removing {len(low_var_cols)} columns with low variance")
            numeric_data = numeric_data.drop(columns=low_var_cols)
        
        # Save a copy of data before advanced processing for potential comparison
        pre_cleaned_data = numeric_data.copy()
        
        # 6. Advanced missing value imputation
        print("\nPerforming advanced missing value imputation...")
        # Define different imputation strategies
        
        # Columns with < 10% missing: Use KNN imputation
        # Columns with >= 10% missing: Use iterative imputation (MICE)
        missing_pct = numeric_data.isnull().mean()
        knn_cols = missing_pct[missing_pct < 0.1].index.tolist()
        iter_cols = missing_pct[missing_pct >= 0.1].index.tolist()
        
        # Create copies for imputation
        if knn_cols:
            knn_data = numeric_data[knn_cols].copy()
            # Apply KNN imputation
            knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
            knn_imputed = pd.DataFrame(
                knn_imputer.fit_transform(knn_data),
                columns=knn_data.columns,
                index=knn_data.index
            )
            numeric_data[knn_cols] = knn_imputed
            print(f"Applied KNN imputation to {len(knn_cols)} columns with <10% missing values")
        
        if iter_cols:
            iter_data = numeric_data[iter_cols].copy()
            # Apply iterative imputation
            iter_imputer = IterativeImputer(max_iter=10, random_state=42)
            iter_imputed = pd.DataFrame(
                iter_imputer.fit_transform(iter_data),
                columns=iter_data.columns,
                index=iter_data.index
            )
            numeric_data[iter_cols] = iter_imputed
            print(f"Applied iterative imputation to {len(iter_cols)} columns with ≥10% missing values")
        
        # 7. Check for and handle outliers using IQR
        print("\nDetecting and handling outliers...")
        
        # Create a heatmap of potential outliers for visualization
        outlier_matrix = pd.DataFrame(index=numeric_data.index)
        
        for col in numeric_data.columns:
            # Calculate IQR
            Q1 = numeric_data[col].quantile(0.25)
            Q3 = numeric_data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            outliers = (numeric_data[col] < lower_bound) | (numeric_data[col] > upper_bound)
            outlier_matrix[col] = outliers.astype(int)
            
            # Count outliers
            outlier_count = outliers.sum()
            outlier_pct = outlier_count / len(numeric_data) * 100
            
            if outlier_count > 0:
                print(f"  {col}: {outlier_count} outliers ({outlier_pct:.2f}%)")
                
                # Handle outliers - cap at the bounds instead of removing
                numeric_data.loc[numeric_data[col] < lower_bound, col] = lower_bound
                numeric_data.loc[numeric_data[col] > upper_bound, col] = upper_bound
        
        # Calculate total outlier percentage by row
        outlier_matrix['total_pct'] = outlier_matrix.sum(axis=1) / len(numeric_data.columns) * 100
        
        # Identify rows with excessive outliers (>50% of features are outliers)
        excessive_outlier_rows = outlier_matrix[outlier_matrix['total_pct'] > 50].index.tolist()
        
        if excessive_outlier_rows:
            print(f"\nIdentified {len(excessive_outlier_rows)} rows with >50% outlier features")
            print("  These rows might be anomalous - recording them for reference")
            
            # Save these rows to a file for further inspection
            if self.id_data is not None:
                anomalous_data = pd.concat([self.id_data.loc[excessive_outlier_rows], 
                                           numeric_data.loc[excessive_outlier_rows]], axis=1)
                anomalous_data.to_csv(os.path.join(self.output_path, 'anomalous_tracts.csv'), index=False)
        
        # 8. Use Isolation Forest as a secondary method to detect outliers
        try:
            print("\nPerforming Isolation Forest outlier detection...")
            iso_forest = IsolationForest(contamination=0.05, random_state=42)
            iso_forest.fit(numeric_data)
            
            # Get anomaly scores
            outlier_scores = iso_forest.decision_function(numeric_data)
            iso_outliers = iso_forest.predict(numeric_data) == -1
            
            print(f"Isolation Forest identified {iso_outliers.sum()} potential outlier rows ({iso_outliers.mean()*100:.2f}%)")
            
            geoid_series = self.id_data["GEOID"].reset_index(drop=True)

            # Save isolation forest outlier scores for reference
            outlier_scores_df = pd.DataFrame({
                'GEOID': geoid_series,
                'Outlier_Score': outlier_scores,
                'Is_Outlier': iso_outliers
            })
            outlier_scores_df.to_csv(os.path.join(self.output_path, 'isolation_forest_outliers.csv'), index=False)
            
            # For very extreme outliers detected by Isolation Forest, apply smoothing
            extreme_outliers_idx = outlier_scores < -0.5  # More negative means more anomalous
            if extreme_outliers_idx.sum() > 0:
                print(f"Applying robust scaling to {extreme_outliers_idx.sum()} extreme outlier rows")
                
                # Apply RobustScaler to these extreme outliers to reduce their impact
                robust_scaler = RobustScaler()
                extreme_data = numeric_data.loc[extreme_outliers_idx].copy()
                extreme_scaled = pd.DataFrame(
                    robust_scaler.fit_transform(extreme_data),
                    columns=extreme_data.columns,
                    index=extreme_data.index
                )
                numeric_data.loc[extreme_outliers_idx] = extreme_scaled
        except Exception as e:
            print(f"Error in Isolation Forest analysis: {e}")
        
        # 9. Confirm data consistency and range
        print("\nVerifying data consistency after preprocessing...")
        
        # Check for any remaining nulls
        remaining_nulls = numeric_data.isnull().sum().sum()
        if remaining_nulls > 0:
            print(f"Warning: {remaining_nulls} null values remain after imputation!")
            print("Applying final median imputation to any remaining nulls")
            numeric_data = numeric_data.fillna(numeric_data.median())
        else:
            print("No null values remain - imputation successful")
        
        # 10. Generate data quality report
        quality_report = pd.DataFrame({
            'Column': numeric_data.columns,
            'Mean': numeric_data.mean(),
            'Median': numeric_data.median(),
            'Std': numeric_data.std(),
            'Min': numeric_data.min(),
            'Max': numeric_data.max(),
            'Skewness': numeric_data.skew(),
            'Kurtosis': numeric_data.kurtosis()
        })
        
        # Save quality report
        quality_report.to_csv(os.path.join(self.output_path, 'data_quality_report.csv'), index=False)
        
        # 11. Create correlation matrix for cleaned data
        correlation = numeric_data.corr()
        
        # Identify extreme correlations (>0.95 or <-0.95)
        extreme_corr_pairs = []
        for i in range(len(correlation.columns)):
            for j in range(i):
                if abs(correlation.iloc[i, j]) > 0.95:
                    extreme_corr_pairs.append({
                        'Feature1': correlation.columns[i],
                        'Feature2': correlation.columns[j],
                        'Correlation': correlation.iloc[i, j]
                    })
        
        if extreme_corr_pairs:
            print(f"\nIdentified {len(extreme_corr_pairs)} pairs of extremely correlated features (|r| > 0.95)")
            
            # Save these for reference
            extreme_corr_df = pd.DataFrame(extreme_corr_pairs)
            extreme_corr_df.to_csv(os.path.join(self.output_path, 'extreme_correlations.csv'), index=False)
            
            # Consider dropping one from each highly correlated pair to reduce multicollinearity
            # (doing this for top correlations only to avoid excessive feature removal)
            if len(extreme_corr_pairs) > 5:
                # Sort by absolute correlation
                extreme_corr_df['abs_corr'] = extreme_corr_df['Correlation'].abs()
                extreme_corr_df = extreme_corr_df.sort_values('abs_corr', ascending=False).drop('abs_corr', axis=1)
                
                # Take top 5 pairs
                top_pairs = extreme_corr_df.head(5)
                
                # For each pair, keep the feature with higher variance
                features_to_drop = []
                for _, row in top_pairs.iterrows():
                    feature1 = row['Feature1']
                    feature2 = row['Feature2']
                    
                    if numeric_data[feature1].var() > numeric_data[feature2].var():
                        features_to_drop.append(feature2)
                    else:
                        features_to_drop.append(feature1)
                
                # Remove duplicates
                features_to_drop = list(set(features_to_drop))
                
                print(f"Removing {len(features_to_drop)} features from highly correlated pairs")
                numeric_data = numeric_data.drop(columns=features_to_drop)
        
        # 12. Final step: Create visualization of data distribution after preprocessing
        try:
            # Visualize data distributions before and after preprocessing
            fig = plt.figure(figsize=(15, 10))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
            
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            
            # Sample a few columns for visualization (up to 5)
            vis_columns = list(pre_cleaned_data.columns[:5])
            
            # Before preprocessing
            pre_cleaned_data[vis_columns].boxplot(ax=ax1)
            ax1.set_title('Data Distribution Before Preprocessing')
            ax1.set_ylabel('Value')
            ax1.set_xticklabels(vis_columns, rotation=45, ha='right')
            
            # After preprocessing
            numeric_data[vis_columns].boxplot(ax=ax2)
            ax2.set_title('Data Distribution After Preprocessing')
            ax2.set_ylabel('Value')
            ax2.set_xticklabels(vis_columns, rotation=45, ha='right')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'preprocessing_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            print(f"Error creating preprocessing visualization: {e}")
        
        # 13. Save final preprocessed dataset
        numeric_data.to_csv(os.path.join(self.output_path, 'preprocessed_data.csv'), index=False)
        
        # Store the final preprocessed data
        self.numeric_data = numeric_data
        
        print(f"\nFinal preprocessed data shape: {self.numeric_data.shape}")
        print(f"Data is now pristine and ready for analysis")
        return True
    
    def perform_statistical_analysis(self):
        """Perform statistical analysis on the preprocessed data."""
        if self.numeric_data is None:
            print("No preprocessed data available. Please preprocess data first.")
            return False
        
        print("\n--- Statistical Analysis ---")
        
        # 1. Basic descriptive statistics
        desc_stats = self.numeric_data.describe().T
        desc_stats['missing'] = self.numeric_data.isnull().sum()
        desc_stats['missing_percent'] = (self.numeric_data.isnull().sum() / len(self.numeric_data)) * 100
        
        # Save descriptive statistics to CSV
        desc_stats.to_csv(os.path.join(self.output_path, 'descriptive_statistics.csv'))
        print("Saved descriptive statistics to CSV")
        
        # 2. Calculate correlation matrix
        correlation = self.numeric_data.corr()
        
        # Save correlation matrix to CSV
        correlation.to_csv(os.path.join(self.output_path, 'correlation_matrix.csv'))
        
        # 3. Identify highly correlated features (absolute correlation > 0.8)
        high_corr = pd.DataFrame(correlation.abs().unstack().sort_values(ascending=False))
        high_corr = high_corr[high_corr[0] > 0.8]
        high_corr = high_corr[high_corr.index.get_level_values(0) != high_corr.index.get_level_values(1)]
        
        # Save high correlations to CSV
        high_corr.to_csv(os.path.join(self.output_path, 'high_correlations.csv'))
        print(f"Identified {len(high_corr)} highly correlated feature pairs")
        
        return True
        
    def perform_feature_selection(self):
        """Select the most important features for clustering."""
        if self.numeric_data is None:
            print("No preprocessed data available. Please preprocess data first.")
            return False
        
        print("\n--- Feature Selection ---")
        
        # 1. Normalize the data for feature selection
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.numeric_data)
        
        # 2. Use Random Forest to select important features
        try:
            # Using a Random Forest to identify important features
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            
            # Create synthetic targets for classification (using KMeans with 5 clusters)
            temp_kmeans = KMeans(n_clusters=5, random_state=42)
            synthetic_targets = temp_kmeans.fit_predict(scaled_data)
            
            # Fit the Random Forest
            rf.fit(scaled_data, synthetic_targets)
            
            # Get feature importances
            importances = rf.feature_importances_
            feature_importances = pd.DataFrame({
                'Feature': self.numeric_data.columns,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Save feature importances to CSV
            feature_importances.to_csv(os.path.join(self.output_path, 'feature_importances.csv'), index=False)
            print(f"Identified and saved importances for {len(feature_importances)} features")
            
            # Store feature importance for later use
            self.feature_importance = feature_importances
            
            # Select top features that account for 80% of the importance
            cumulative_importance = np.cumsum(feature_importances['Importance'])
            threshold_idx = np.where(cumulative_importance >= 0.8)[0][0]
            selected_features = feature_importances.iloc[:threshold_idx+1]['Feature'].tolist()
            
            print(f"Selected {len(selected_features)} features that account for 80% of importance")
            
            # Update numeric data to include only selected features
            self.numeric_data = self.numeric_data[selected_features]
            return True
            
        except Exception as e:
            print(f"Error in feature selection: {e}")
            # Fallback: Just use all features if selection fails
            print("Using all available features")
            return True
    
    def perform_dimensionality_reduction(self):
        """Perform PCA for dimensionality reduction."""
        if self.numeric_data is None:
            print("No preprocessed data available. Please preprocess data first.")
            return False
        
        print("\n--- Dimensionality Reduction ---")
        
        # 1. Normalize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.numeric_data)
        
        # 2. Perform selected dimensionality reduction methods for comparison
        for method in DR_METHODS:
            if method == "pca":
                try:
                    pca = PCA()
                    pca_result = pca.fit_transform(scaled_data)
                    explained_variance = pca.explained_variance_ratio_
                    cumulative_variance = np.cumsum(explained_variance)
                    n_components = np.where(cumulative_variance >= 0.8)[0][0] + 1
                    print(f"PCA: {n_components} components explain >80% of variance")
                    pca_df = pd.DataFrame(
                        pca_result[:, :n_components],
                        columns=[f'PC{i+1}' for i in range(n_components)]
                    )
                    components_df = pd.DataFrame(
                        pca.components_.T[:, :n_components],
                        columns=[f'PC{i+1}' for i in range(n_components)],
                        index=self.numeric_data.columns
                    )
                    components_df.to_csv(os.path.join(self.output_path, 'pca_components.csv'))
                    self.pca_results = pca_df
                except Exception as e:
                    print(f"Error in PCA: {e}")
            elif method == "umap":
                if umap is None:
                    print("umap not installed; skipping UMAP")
                    continue
                try:
                    reducer = umap.UMAP(random_state=42)
                    umap_result = reducer.fit_transform(scaled_data)
                    df_umap = pd.DataFrame(
                        umap_result,
                        columns=[f'UMAP{i+1}' for i in range(umap_result.shape[1])]
                    )
                    df_umap.to_csv(os.path.join(self.output_path, 'umap_components.csv'), index=False)
                    self.umap_results = df_umap
                    print(f"UMAP: reduced to {umap_result.shape[1]} dimensions")
                except Exception as e:
                    print(f"Error in UMAP: {e}")
            else:
                print(f"Unknown DR method: {method}")
        return True
    
    def perform_clustering(self, max_clusters=7):
        """
        Perform KMeans clustering to segment census tracts.
        Uses the Elbow method and Silhouette score to determine optimal cluster count.
        
        Returns dict with optimal cluster information and scaled data for modeling.
        """
        if self.numeric_data is None:
            print("No preprocessed data available. Please preprocess data first.")
            return False
        
        print("\n--- Clustering Analysis ---")
        
        # 1. Normalize the data for clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(self.numeric_data)
        
        # Store scaled data for later use in XGBoost
        self.scaled_data = scaled_data
        
        # 2. Determine optimal number of clusters using Elbow method and Silhouette score
        wcss = []  # Within-cluster sum of squares
        silhouette_scores = []
        cluster_models = {}  # Store models for each cluster count
        cluster_assignments = {}  # Store cluster assignments for each count
        
        # Start from 2 clusters
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            # Apply KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(scaled_data)
            
            # Store model and assignments
            cluster_models[n_clusters] = kmeans
            cluster_assignments[n_clusters] = kmeans.labels_
            
            # Calculate WCSS
            wcss.append(kmeans.inertia_)
            
            # Calculate silhouette score
            cluster_labels = kmeans.labels_
            try:
                silhouette_avg = silhouette_score(scaled_data, cluster_labels)
                silhouette_scores.append(silhouette_avg)
                print(f"For n_clusters = {n_clusters}, the silhouette score is {silhouette_avg:.3f}")
            except:
                silhouette_scores.append(0)
        
        # Find optimal number of clusters
        # Method 1: Largest silhouette score
        optimal_clusters_silhouette = cluster_range[silhouette_scores.index(max(silhouette_scores))]
        
        # Method 2: Elbow method (find point of diminishing returns)
        # Approximate the "elbow" point using rate of change
        wcss_diff = np.diff(wcss)
        wcss_diff2 = np.diff(wcss_diff)  # Second derivative
        elbow_idx = np.argmax(wcss_diff2) + 1 if len(wcss_diff2) > 0 else 0
        optimal_clusters_elbow = cluster_range[elbow_idx] if elbow_idx < len(cluster_range) else cluster_range[0]
        
        # Use silhouette score method as primary, with elbow as fallback
        self.optimal_clusters = optimal_clusters_silhouette
        print(f"Optimal number of clusters: {self.optimal_clusters} (silhouette method)")
        print(f"Alternative by elbow method: {optimal_clusters_elbow}")
        
        # Store all clustering information for later use
        self.clustering_info = {
            'optimal_clusters': self.optimal_clusters,
            'alternative_clusters': optimal_clusters_elbow,
            'silhouette_scores': silhouette_scores,
            'wcss': wcss,
            'cluster_models': cluster_models,
            'cluster_assignments': cluster_assignments,
            'cluster_range': list(cluster_range)
        }
        
        # Define the specific cluster counts we want to use
        desired_cluster_counts = CLUSTER_COUNTS  # Manually specified cluster counts
        
        # Make sure we have models for all our desired cluster counts
        for n_clusters in desired_cluster_counts:
            if n_clusters not in cluster_models:
                print(f"Creating KMeans model with {n_clusters} clusters...")
                kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans_model.fit(scaled_data)
                cluster_models[n_clusters] = kmeans_model
                cluster_assignments[n_clusters] = kmeans_model.labels_
            
        # Use 10 clusters for initial visualization and analysis
        self._apply_clustering(DEFAULT_CLUSTER_COUNT, scaled_data)
        
        # Train XGBoost with multiple cluster counts as a grid search parameter
        # Use our predefined cluster counts
        cluster_range_for_grid = desired_cluster_counts  # Use the same cluster counts we created models for
        
        print(f"Will evaluate {len(cluster_range_for_grid)} different cluster counts in XGBoost grid search: {cluster_range_for_grid}")
        
        # Train the model with multiple cluster counts as a hyperparameter
        self.train_xgboost_model(scaled_data, cluster_assignments, cluster_range_for_grid)
        
        print(f"Saved clustering results to CSV and GeoJSON")
        return True
    
    def _apply_clustering(self, n_clusters, scaled_data):
        """
        Helper method to apply a specific clustering result to the data.
        Uses pre-computed cluster assignments.
        """
        # Get the labels for this cluster count
        cluster_labels = self.clustering_info['cluster_assignments'][n_clusters]
        
        # Save the cluster labels for modeling
        self.cluster_labels = cluster_labels
        
        # Add cluster labels to the original data
        self.cluster_results = self.data.copy()
        self.cluster_results['cluster'] = cluster_labels
        
        # Also add to the geo_data
        self.geo_data['cluster'] = cluster_labels
        
        # Analyze clusters
        cluster_analysis = pd.DataFrame()
        for cluster in range(n_clusters):
            cluster_data = self.numeric_data[self.cluster_results['cluster'] == cluster]
            cluster_analysis[f'Cluster {cluster}'] = cluster_data.mean()
        
        # Save cluster analysis
        cluster_analysis.to_csv(os.path.join(self.output_path, f'cluster_profiles_{n_clusters}.csv'))
        print(f"Saved cluster profiles for {n_clusters} clusters to CSV")
        
        # Save results to CSV and GeoJSON for mapping
        self.geo_data.to_file(os.path.join(self.output_path, f'clustered_tracts_{n_clusters}.geojson'), driver='GeoJSON')
        
        # Extract just the cluster and identifier columns for a lightweight CSV
        result_df = self.cluster_results[['GEOID', 'STATEFP', 'COUNTYFP', 'TRACTCE', 'cluster']].copy()
        result_df.to_csv(os.path.join(self.output_path, f'tract_clusters_{n_clusters}.csv'), index=False)
    
    def find_doppelgangers(self, n_doppelgangers=10):
        """
        Find census tracts that are similar based on non-geographic characteristics 
        but located in different geographic areas.
        """
        if self.cluster_results is None:
            print("No clustering results available. Please perform clustering first.")
            return False
        
        print("\n--- Finding Doppelganger Census Tracts ---")
        
        # 1. Group by cluster and state
        cluster_state_counts = self.cluster_results.groupby(['cluster', 'STATEFP']).size().reset_index(name='count')
        
        # 2. Find clusters that span multiple states
        multi_state_clusters = cluster_state_counts.groupby('cluster').size()
        multi_state_clusters = multi_state_clusters[multi_state_clusters > 1].index.tolist()
        
        print(f"Found {len(multi_state_clusters)} clusters that span multiple states")
        
        # Use tqdm for progress tracking if available
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            multi_state_clusters_iter = tqdm(multi_state_clusters, desc="Processing clusters for doppelgangers")
        else:
            multi_state_clusters_iter = multi_state_clusters
        
        if not multi_state_clusters:
            print("No doppelgangers found (no clusters span multiple states)")
            return False
        
        # 3. For each multi-state cluster, find representative tracts
        doppelgangers = []
        
        for cluster in multi_state_clusters_iter:
            # Get tracts in this cluster
            cluster_tracts = self.cluster_results[self.cluster_results['cluster'] == cluster]
            
            # Group by state
            states = cluster_tracts['STATEFP'].unique()
            
            # If this cluster has tracts in multiple states
            if len(states) >= 2:
                # For each state pair, find doppelgangers
                for i in range(len(states)):
                    for j in range(i+1, len(states)):
                        state1 = states[i]
                        state2 = states[j]
                        
                        # Get tracts from each state
                        state1_tracts = cluster_tracts[cluster_tracts['STATEFP'] == state1]
                        state2_tracts = cluster_tracts[cluster_tracts['STATEFP'] == state2]
                        
                        # Sample a few tracts from each state (or take all if less than n_doppelgangers)
                        sample_size = min(n_doppelgangers, len(state1_tracts), len(state2_tracts))
                        
                        if sample_size > 0:
                            state1_sample = state1_tracts.sample(sample_size, random_state=42)
                            state2_sample = state2_tracts.sample(sample_size, random_state=42)
                            
                            # Combine samples
                            for idx in range(sample_size):
                                doppelgangers.append({
                                    'cluster': cluster,
                                    'tract1_geoid': state1_sample.iloc[idx]['GEOID'],
                                    'tract1_name': state1_sample.iloc[idx]['NAME'],
                                    'tract1_state': state1,
                                    'tract2_geoid': state2_sample.iloc[idx]['GEOID'],
                                    'tract2_name': state2_sample.iloc[idx]['NAME'],
                                    'tract2_state': state2
                                })
        
        # Create dataframe of doppelgangers
        doppelgangers_df = pd.DataFrame(doppelgangers)
        
        # Save doppelgangers to CSV
        if len(doppelgangers) > 0:
            doppelgangers_df.to_csv(os.path.join(self.output_path, 'doppelganger_tracts.csv'), index=False)
            print(f"Found and saved {len(doppelgangers)} doppelganger pairs")
            # Sampling-based done
        else:
            print("No sampling-based doppelgangers found")
        
        # NearestNeighbors-based doppelganger search
        if "nearest_neighbors" in DOPPELGANGER_METHODS:
            print("\n--- Finding Doppelganger via NearestNeighbors ---")
            nbrs = NearestNeighbors(n_neighbors=n_doppelgangers+1, metric=SIMILARITY_METRIC)
            nbrs.fit(self.numeric_data.values)
            distances, indices = nbrs.kneighbors(self.numeric_data.values)
            similarities = 1 - distances
            nn_pairs = []
            for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
                for dist, j in zip(dist_row[1:], idx_row[1:]):
                    sim = 1 - dist
                    if sim >= DOPPELGANGER_SIMILARITY_THRESHOLD and self.cluster_results.loc[i, 'STATEFP'] != self.cluster_results.loc[j, 'STATEFP']:
                        nn_pairs.append({
                            'cluster': self.cluster_results.loc[i, 'cluster'],
                            'tract1_geoid': self.cluster_results.loc[i, 'GEOID'],
                            'tract1_state': self.cluster_results.loc[i, 'STATEFP'],
                            'tract2_geoid': self.cluster_results.loc[j, 'GEOID'],
                            'tract2_state': self.cluster_results.loc[j, 'STATEFP'],
                            'similarity': float(sim),
                        })
            nn_df = pd.DataFrame(nn_pairs).sort_values('similarity', ascending=False)
            nn_df.to_csv(os.path.join(self.output_path, 'doppelganger_nn.csv'), index=False)
            print(f"Saved {len(nn_df)} NearestNeighbors-based pairs")
        return True
    
    def find_precise_doppelgangers(self) -> bool:
        """Find doppelgangers using direct, weighted similarity for high-accuracy matching."""
        if self.numeric_data is None or self.cluster_results is None:
            print("No data or clustering results. Please run preprocessing and clustering first.")
            return False
        print("\n--- Finding Precise Doppelganger Tracts ---")
        # Scale data
        scaler = StandardScaler()
        scaled = scaler.fit_transform(self.numeric_data)
        # Apply feature weights
        weights = np.array([FEATURE_WEIGHTS.get(col, FEATURE_WEIGHT_DEFAULT)
                            for col in self.numeric_data.columns])
        weighted = scaled * weights
        weighted_df = pd.DataFrame(weighted, index=self.numeric_data.index)
        # Identify multi-state clusters
        cluster_state_counts = self.cluster_results.groupby(['cluster', 'STATEFP']).size().reset_index(name='count')
        multi_state = cluster_state_counts.groupby('cluster').size()
        multi_state = multi_state[multi_state > 1].index.tolist()
        all_pairs = []
        for cluster in multi_state:
            subset = self.cluster_results[self.cluster_results['cluster'] == cluster]
            states = subset['STATEFP'].unique()
            for i in range(len(states)):
                for j in range(i+1, len(states)):
                    s1, s2 = states[i], states[j]
                    idx1 = subset[subset['STATEFP']==s1].index
                    idx2 = subset[subset['STATEFP']==s2].index
                    feat1 = weighted_df.loc[idx1].values
                    feat2 = weighted_df.loc[idx2].values
                    sim = cosine_similarity(feat1, feat2)
                    for m, tract1 in enumerate(idx1):
                        top_idx = np.argsort(-sim[m])[:DOPPELGANGER_SAMPLE_SIZE]
                        for n in top_idx:
                            score = sim[m, n]
                            if score >= DOPPELGANGER_SIMILARITY_THRESHOLD:
                                all_pairs.append({
                                    'cluster': cluster,
                                    'tract1_geoid': tract1,
                                    'tract1_state': s1,
                                    'tract2_geoid': idx2[n],
                                    'tract2_state': s2,
                                    'similarity_score': score
                                })
        df = pd.DataFrame(all_pairs)
        if not df.empty:
            df.to_csv(os.path.join(self.output_path, 'precise_doppelganger_tracts.csv'), index=False)
            # Evaluation metrics
            mean_sim = df['similarity_score'].mean()
            min_sim = df['similarity_score'].min()
            max_sim = df['similarity_score'].max()
            summary = pd.DataFrame([{
                'mean_similarity': mean_sim,
                'min_similarity': min_sim,
                'max_similarity': max_sim,
                'num_pairs': len(df)
            }])
            summary.to_csv(os.path.join(self.output_path, 'doppelganger_similarity_summary.csv'), index=False)
            print(f"Found {len(df)} precise doppelganger pairs (mean sim: {mean_sim:.3f})")
            return True
        print("No precise doppelgangers found")
        return False
    
    def train_xgboost_model(self, X, y_dict, cluster_range):
        """
        Train an XGBoost model to predict clusters using grid search for hyperparameter optimization.
        Tests multiple optimizers including Adam and utilizes GPU acceleration if available.
        Tests different cluster counts separately to find the optimal configuration.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix (scaled data)
        y_dict : dict
            Dictionary of cluster labels from KMeans, keyed by number of clusters
        cluster_range : list
            List of cluster counts to test in the grid search
        """
        print("\n--- XGBoost Classification with Grid Search ---")
        
        # We'll use the first cluster count for initial data splitting
        # The actual model will be trained with different cluster counts
        first_cluster_count = cluster_range[0]
        primary_labels = y_dict[first_cluster_count]
        
        # Split data using the primary clustering as stratification guide
        X_train, X_test, y_train_idx, y_test_idx = train_test_split(
            X, np.arange(len(primary_labels)), test_size=0.3, random_state=42, stratify=primary_labels
        )
        
        # Create training and test dictionaries for each cluster count
        self.y_train_dict = {}
        self.y_test_dict = {}
        
        for n_clusters in cluster_range:
            cluster_labels = y_dict[n_clusters]
            self.y_train_dict[n_clusters] = cluster_labels[y_train_idx]
            self.y_test_dict[n_clusters] = cluster_labels[y_test_idx]
        
        print(f"Training XGBoost classifier with {len(cluster_range)} different cluster counts: {cluster_range}")
        
        # Define parameter grid for grid search (excluding cluster counts)
        # Testing different optimizers and hyperparameters
        param_grid = {
            'learning_rate': [0.01, 0.2],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200],
            'reg_alpha': [0, 1],
            'reg_lambda': [0, 10],
            'subsample': [0.6, 0.8],
            'colsample_bytree': [0.6, 0.8],
        }
        
        # Add GPU-specific parameters if CUDA is available
        if self.use_gpu and self.cuda_available:
            # For GPU, we'll use fixed parameters rather than grid search options to maximize GPU efficiency
            param_grid['tree_method'] = ['gpu_hist']
            print("Using CUDA GPU acceleration for XGBoost")
        else:
            # CPU-only methods - more options for grid search
            param_grid['tree_method'] = ['auto', 'exact', 'approx', 'hist']
            print("Using CPU for XGBoost computation")
        
        # Create separate models for each cluster count
        models = {}
        scores = {}
        best_params_by_cluster = {}
        
        # Configure number of jobs based on whether GPU is used
        # With GPU, we don't want to use multiple workers as they'll compete for GPU resources
        n_jobs = 1 if (self.use_gpu and self.cuda_available) else -1
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Start timing the grid search
        start_time = time()
        
        # Calculate total parameter combinations for progress tracking
        param_combinations = 1
        for param_values in param_grid.values():
            param_combinations *= len(param_values)
        
        print(f"Grid search will evaluate {param_combinations} parameter combinations for each of {len(cluster_range)} cluster counts")
        print(f"Using {cv.get_n_splits()} cross-validation folds")
        total_fits = param_combinations * cv.get_n_splits() * len(cluster_range)
        print(f"Total of {total_fits} model fits will be performed")
        
        # Define XGBoost callback for training progress if tqdm is available
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            class TqdmProgressCallback(xgb.callback.TrainingCallback):
                def __init__(self, max_rounds=100, show_interval=10):
                    self.pbar = None
                    self.max_rounds = max_rounds
                    self.show_interval = show_interval
                    
                def before_training(self, model):
                    self.pbar = tqdm(total=self.max_rounds, desc="XGBoost Training", leave=False)
                    return model
                    
                def after_iteration(self, model, epoch, evals_log):
                    self.pbar.update(1)
                    return False  # Continue training
                    
                def after_training(self, model):
                    self.pbar.close()
                    return model
        
        # Setup progress bars for outer loop (cluster counts)
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            outer_pbar = tqdm(total=len(cluster_range), desc="Cluster Count Progress")
            inner_pbar = tqdm(total=param_combinations * cv.get_n_splits(), desc="Grid Search Progress", leave=False)
        
        # Process each cluster count separately
        for n_clusters in cluster_range:
            print(f"\nTraining models for {n_clusters} clusters")
            
            # Create XGBoost classifier with correct number of classes
            current_params = {
                'random_state': 42, 
                'eval_metric': 'mlogloss',
                'objective': 'multi:softprob',
                'num_class': n_clusters
            }
            
            # Add GPU parameters if available
            if self.use_gpu and self.cuda_available:
                current_params['tree_method'] = 'gpu_hist'
                current_params['gpu_id'] = 0
                current_params['predictor'] = 'gpu_predictor'
            
            xgb_model = XGBClassifier(**current_params)
            
            # Add callbacks for this model if using GPU and progress tracking
            if TQDM_AVAILABLE and USE_PROGRESS_BARS and self.use_gpu and self.cuda_available:
                xgb_model.set_params(callbacks=[TqdmProgressCallback(200)])  # Assume max of 200 rounds
            
            # Create grid search for this cluster count
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=cv,
                scoring='f1_weighted',
                n_jobs=n_jobs,  # Number of parallel jobs
                verbose=1 if not (TQDM_AVAILABLE and USE_PROGRESS_BARS) else 0  # Only use verbose if tqdm not available
            )
            
            # Fit grid search for this cluster count
            try:
                if TQDM_AVAILABLE and USE_PROGRESS_BARS:
                    # Reset inner progress bar
                    inner_pbar.reset()
                    inner_pbar.set_description(f"Grid Search for {n_clusters} clusters")
                    
                    # Monkey patch the fit method to track progress
                    original_fit = xgb_model.fit
                    
                    def fit_with_progress(*args, **kwargs):
                        result = original_fit(*args, **kwargs)
                        inner_pbar.update(1)
                        return result
                    
                    # Apply the monkey patch
                    try:
                        xgb_model.fit = fit_with_progress
                        grid_search.fit(X_train, self.y_train_dict[n_clusters])
                    finally:
                        # Restore original method
                        xgb_model.fit = original_fit
                else:
                    grid_search.fit(X_train, self.y_train_dict[n_clusters])
                
                # Store best model and score for this cluster count
                models[n_clusters] = grid_search.best_estimator_
                scores[n_clusters] = grid_search.best_score_
                best_params_by_cluster[n_clusters] = grid_search.best_params_
                
                print(f"Best score for {n_clusters} clusters: {grid_search.best_score_:.4f}")
                print(f"Best parameters: {grid_search.best_params_}")
                
            except Exception as e:
                print(f"Error training model for {n_clusters} clusters: {e}")
                continue
                
            if TQDM_AVAILABLE and USE_PROGRESS_BARS:
                outer_pbar.update(1)
        
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            outer_pbar.close()
            inner_pbar.close()
        
        # Report grid search time and results
        grid_search_time = time() - start_time
        print(f"\nGrid search completed in {grid_search_time:.2f} seconds")
        
        # Find the best cluster count based on cross-validation scores
        if scores:
            best_cluster_count = max(scores, key=scores.get)
            best_score = scores[best_cluster_count]
            print(f"Best overall cluster count: {best_cluster_count} with score: {best_score:.4f}")
            
            # Save the best model and parameters
            self.best_model = models[best_cluster_count]
            self.best_params = best_params_by_cluster[best_cluster_count]
            self.best_params['num_clusters'] = best_cluster_count
            
            print("\nBest parameters found:")
            for param, value in self.best_params.items():
                print(f"  {param}: {value}")
            
            # Now evaluate with the test labels for the best cluster count
            y_test = self.y_test_dict[best_cluster_count]
            y_pred = self.best_model.predict(X_test)
        else:
            print("No successful models were trained. Using default cluster count.")
            best_cluster_count = self.optimal_clusters
            y_test = self.y_test_dict[best_cluster_count]
            y_pred = None
            
        # Calculate metrics if we have predictions
        if y_pred is not None:
            try:
                # Ensure both are integer arrays with same format
                y_test_int = y_test.astype(int) if hasattr(y_test, 'astype') else np.array(y_test, dtype=int)
                y_pred_int = y_pred.astype(int) if hasattr(y_pred, 'astype') else np.array(y_pred, dtype=int)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_int, y_pred_int)
                f1 = f1_score(y_test_int, y_pred_int, average='weighted')
                
                print(f"\nModel Evaluation Metrics:")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  F1 Score (weighted): {f1:.4f}")
                print(f"  Cluster Count: {best_cluster_count}")
                
                # Store metrics
                self.evaluation_metrics = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'best_params': self.best_params
                }
            except Exception as e:
                print(f"Error calculating metrics: {e}")
                self.evaluation_metrics = {
                    'error': f"Error calculating metrics: {e}",
                    'best_params': self.best_params
                }
        else:
            print(f"\nNo model evaluation metrics available")
            self.evaluation_metrics = {
                'error': 'No successful model training',
                'best_cluster_count': best_cluster_count
            }
            
        # Apply the best cluster count to update our data
        # This ensures all results are based on the optimal cluster count
        if y_pred is not None:
            # Apply the best clustering to the data
            self._apply_clustering(best_cluster_count, X)
        
            # Compare all cluster counts
            print("\nScores by cluster count:")
            for n_clusters in sorted(scores.keys()):
                print(f"  {n_clusters} clusters: {scores[n_clusters]:.4f}")
        
            # Save detailed classification report
            try:
                # Convert labels to ensure they have consistent format
                y_test_int = y_test.astype(int) if hasattr(y_test, 'astype') else np.array(y_test, dtype=int)
                y_pred_int = y_pred.astype(int) if hasattr(y_pred, 'astype') else np.array(y_pred, dtype=int)
                
                class_report = classification_report(y_test_int, y_pred_int, output_dict=True)
                report_df = pd.DataFrame(class_report).transpose()
                report_df.to_csv(os.path.join(self.output_path, f"classification_report_{best_cluster_count}_clusters.csv"))
                print("Classification report saved successfully.")
            except Exception as e:
                print(f"Error generating classification report: {e}")
            # Save confusion matrix
            try:
                y_test_int = y_test.astype(int) if hasattr(y_test, 'astype') else np.array(y_test, dtype=int)
                y_pred_int = y_pred.astype(int) if hasattr(y_pred, 'astype') else np.array(y_pred, dtype=int)
                
                conf_matrix = confusion_matrix(y_test_int, y_pred_int)
                np.savetxt(os.path.join(self.output_path, f"confusion_matrix_{best_cluster_count}_clusters.csv"), 
                          conf_matrix, delimiter=',', fmt='%d')
                print("Confusion matrix saved successfully.")
            except Exception as e:
                print(f"Error generating confusion matrix: {e}")
            
            # Save feature importance if using a tree-based model
            try:
                feature_importance = self.best_model.feature_importances_
                # Convert to DataFrame for better visualization
                importance_df = pd.DataFrame({
                    'Feature': range(X.shape[1]),  # Feature indices as we don't have names
                    'Importance': feature_importance
                })
                importance_df = importance_df.sort_values('Importance', ascending=False)
                importance_df.to_csv(os.path.join(self.output_path, "feature_importance.csv"), index=False)
                
                # Plot feature importance
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
                plt.title(f'Top 20 Feature Importance - {best_cluster_count} clusters')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, "feature_importance.png"), dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                print(f"Could not save feature importance: {e}")
                
            # Save model
            try:
                model_json_path = os.path.join(self.output_path, 'xgboost_model.json')
                model_binary_path = os.path.join(self.output_path, 'xgboost_model.bin')
                
                self.best_model.save_model(model_json_path)
                print(f"XGBoost model saved to: {model_json_path}")
                
                # Also save in binary format
                self.best_model.save_model(model_binary_path)
                print(f"XGBoost model also saved to: {model_binary_path}")
                
                # Save model configuration separately
                model_config = {
                    'best_params': self.best_params,
                    'gpu_enabled': self.use_gpu,
                    'cuda_available': self.cuda_available,
                    'num_features': X.shape[1],
                    'optimal_clusters': self.best_params['num_clusters'],
                    'tested_cluster_counts': cluster_range
                }
                
                pd.DataFrame([model_config]).to_csv(
                    os.path.join(self.output_path, 'model_config.csv'), index=False
                )
            except Exception as e:
                print(f"Error saving model: {e}")
        
        print("XGBoost training and evaluation complete")
        return True
        
    def analyze_feature_clusters(self):
        """
        Analyze how features contribute to different clusters using the trained XGBoost model.
        """
        if self.best_model is None or self.numeric_data is None:
            print("No model or data available. Please train model first.")
            return False
            
        print("\n--- Feature-Cluster Analysis ---")
        
        # Get feature names
        feature_names = self.numeric_data.columns.tolist()
        
        # Create feature importance for each cluster using SHAP if available
        try:
            import shap
            print("Using SHAP for feature importance analysis...")
            
            # Create explainer
            explainer = shap.TreeExplainer(self.best_model)
            
            # Need to ensure we're using the same features the model was trained on
            print(f"Model trained with {self.best_model.n_features_in_} features")
            print(f"Current feature set has {len(feature_names)} features")
            
            # Only select the subset of features the model was trained on
            if hasattr(self.best_model, 'n_features_in_'):
                if len(feature_names) > self.best_model.n_features_in_:
                    feature_names = feature_names[:self.best_model.n_features_in_]
                    print(f"Using first {len(feature_names)} features for SHAP analysis")
            
            # Sample data for SHAP analysis (use a subset if dataset is large)
            sample_data = self.numeric_data[feature_names].sample(
                min(1000, len(self.numeric_data)), random_state=42
            )
            X_sample = StandardScaler().fit_transform(sample_data)
            
            try:
                # Calculate SHAP values
                shap_values = explainer.shap_values(X_sample)
                
                # Create and save SHAP summary plot
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_path, 'shap_summary.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
                # Create cluster-specific SHAP analysis
                for cluster in range(len(shap_values)):
                    plt.figure(figsize=(12, 8))
                    shap.summary_plot(
                        shap_values[cluster], X_sample, feature_names=feature_names, 
                        show=False, plot_type='bar'
                    )
                    plt.title(f"Feature Importance for Cluster {cluster}")
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(self.output_path, f'shap_cluster_{cluster}.png'), 
                        dpi=300, bbox_inches='tight'
                    )
                    plt.close()
                    
                print("SHAP analysis complete. Plots saved to output directory.")
            except Exception as e:
                print(f"Error during SHAP visualization: {e}")
                print("Falling back to standard feature importance analysis")
            
        except ImportError:
            print("SHAP package not available. Using standard feature importance analysis.")
            
            # Use built-in feature importance
            feature_importance = self.best_model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title("Top 20 Feature Importances")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, 'feature_importance.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        return True
        
    def run_full_analysis(self) -> bool:
        """Run the complete analysis pipeline.
        
        This method executes the full tract segmentation workflow, including:
        - Data loading and preprocessing
        - Statistical analysis
        - Feature selection
        - Dimensionality reduction
        - Clustering
        - XGBoost model training with hyperparameter optimization
        - Feature-cluster relationship analysis
        - Doppelganger identification
        
        Returns:
            bool: True if the full analysis completed successfully, False otherwise
        """
        print("Starting full tract segmentation analysis...")
        
        # Track timing for performance analysis
        start_time: float = time()
        
        # Create progress tracking for overall workflow if tqdm is available
        pipeline_steps = [
            "Data Loading", 
            "Preprocessing", 
            "Statistical Analysis", 
            "Feature Selection",
            "Dimensionality Reduction", 
            "Clustering", 
            "Feature-Cluster Analysis", 
            "Doppelganger Identification"
        ]
        
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            workflow_progress = tqdm(total=len(pipeline_steps), desc="Analysis Pipeline Progress")
        
        # 1. Load the data
        if not self.load_data():
            print("ERROR: Data loading failed. Analysis aborted.")
            return False
        
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            workflow_progress.update(1)
            workflow_progress.set_description(f"Completed: {pipeline_steps[0]}")
        
        # 2. Preprocess the data
        if not self.preprocess_data():
            print("ERROR: Data preprocessing failed. Analysis aborted.")
            return False
            
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            workflow_progress.update(1)
            workflow_progress.set_description(f"Completed: {pipeline_steps[1]}")
        
        # 3. Perform statistical analysis
        result = self.perform_statistical_analysis()
        if not result:
            print("WARNING: Statistical analysis encountered issues. Continuing with analysis.")
            
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            workflow_progress.update(1)
            workflow_progress.set_description(f"Completed: {pipeline_steps[2]}")
        
        # 4. Perform feature selection
        result = self.perform_feature_selection()
        if not result:
            print("WARNING: Feature selection encountered issues. Continuing with analysis.")
            
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            workflow_progress.update(1)
            workflow_progress.set_description(f"Completed: {pipeline_steps[3]}")
        
        # 5. Perform dimensionality reduction
        result = self.perform_dimensionality_reduction()
        if not result:
            print("WARNING: Dimensionality reduction encountered issues. Continuing with analysis.")
            
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            workflow_progress.update(1)
            workflow_progress.set_description(f"Completed: {pipeline_steps[4]}")
        
        # 6. Perform clustering
        if not self.perform_clustering():
            print("ERROR: Clustering failed. Analysis aborted.")
            if TQDM_AVAILABLE and USE_PROGRESS_BARS:
                workflow_progress.close()
            return False
            
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            workflow_progress.update(1)
            workflow_progress.set_description(f"Completed: {pipeline_steps[5]}")
        
        # 7. Analyze feature-cluster relationships
        result = self.analyze_feature_clusters()
        if not result:
            print("WARNING: Feature-cluster analysis encountered issues. Continuing with analysis.")
            
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            workflow_progress.update(1)
            workflow_progress.set_description(f"Completed: {pipeline_steps[6]}")
        
        # 8. Find doppelgangers
        result = self.find_doppelgangers()
        if not result:
            print("WARNING: Doppelganger identification encountered issues. Continuing with analysis.")
            
        if TQDM_AVAILABLE and USE_PROGRESS_BARS:
            workflow_progress.update(1)
            workflow_progress.set_description(f"Completed: {pipeline_steps[7]}")
            workflow_progress.close()
        
        # Calculate total runtime
        end_time: float = time()
        
        total_runtime: float = end_time - start_time
        
        runtime_minutes: float = total_runtime / 60
        
        # Save analysis summary
        summary: Dict[str, Any] = {
            
            "total_runtime_seconds": total_runtime,
            
            "total_runtime_minutes": runtime_minutes,
            
            "num_input_features": len(self.data.columns) if self.data is not None else 0,
            
            "num_selected_features": len(self.numeric_data.columns) if self.numeric_data is not None else 0,
            
            "num_observations": len(self.data) if self.data is not None else 0,
            
            "num_clusters": len(np.unique(self.cluster_labels)) if self.cluster_labels is not None else 0,
            
            "model_performance": self.evaluation_metrics
        
        }
        
        # Save summary to file
        summary_df = pd.DataFrame([summary])
        
        summary_df.to_csv(os.path.join(self.output_path, "analysis_summary.csv"), index=False)
        
        print(f"\nAnalysis complete in {runtime_minutes:.2f} minutes!")
        
        print(f"Results saved to: {self.output_path}")
        
        return True


def main() -> None:
    """Main function to run the tract segmentation analysis.
    
    This function creates an instance of the TractSegmentationAnalysis class 
    and executes the full analysis pipeline. It will use GPU acceleration if available.
    """
    # Print header
    print("Census Tract Segmentation Analysis")
    
    print("===================================")
    
    # Check CUDA availability before creating the analyzer
    # This was already done during module initialization but we log it again here
    print(f"GPU Status: {'Available with CUDA' if CUDA_AVAILABLE else 'Not available or CUDA not detected'}")
    
    # Create analyzer with GPU enabled if available
    analyzer = TractSegmentationAnalysis(use_gpu=True)  # Will fall back to CPU if GPU not available
    
    success = analyzer.run_full_analysis()
    
    # Return appropriate exit code
    if not success:
        
        print("\nAnalysis completed with errors.")
        
        import sys
        
        sys.exit(1)
    


if __name__ == "__main__":
    
    main()