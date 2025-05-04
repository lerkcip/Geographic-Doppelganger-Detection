# Census Data Files

Due to GitHub file size limitations, the complete census data files are not included directly in this repository.

## Census Data Sources

This project relies on several data sources from the U.S. Census Bureau and other organizations:

1. American Community Survey (ACS) data
2. Census tract shapefiles
3. Additional demographic and economic datasets

## How to Obtain These Files

### Option 1: Direct Download from Source

The census data used in this project can be downloaded from the following sources:
- [U.S. Census Bureau](https://www.census.gov/data.html)
- [American Community Survey (ACS)](https://www.census.gov/programs-surveys/acs/data.html)
- [TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html)

### Option 2: Contact Repository Owner

For access to the exact datasets used in this project, please contact the repository owner.

## Data Directory Structure

When downloaded, the data should be organized as follows:

```
data/
├── Census Data/
│   └── DSC630 Data Files/
│       └── [Various census data files]
```

## Processing the Data

Once you have obtained the data, run the data processing scripts in the `/code` directory:
- `extract_census_data.py` - Extract and preprocess census data
- `integrated_tract_data.py` - Integrate multiple data sources
