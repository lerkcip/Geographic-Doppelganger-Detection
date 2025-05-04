# Geographic Doppelganger Detection via Integrated Census Tract Clustering

A data-driven approach to identifying similar neighborhoods across the United States using census tract data and machine learning techniques.

## Project Overview

This repository contains code, data, and documentation for a geographic area similarity analysis system. The project combines:

1. Census tract data from multiple sources
2. Machine learning segmentation techniques
3. Sophisticated visualization methods
4. Statistical analysis of demographic patterns

The goal is to identify "geographic doppelgangers" - census tracts across the United States that share similar demographic, economic, and social characteristics despite being geographically distant.

## Repository Structure

- `/code`: Python scripts for data processing, analysis, and segmentation
- `/notebooks`: Jupyter notebooks for analysis and visualization
- `/data`: Data directory (see note about large data files below)
- `/figures`: Visualizations and result plots
- `/docs`: Documentation, white papers, and presentation materials
- `/output`: Generated output files and results

## Important Note About Large Files

Due to GitHub's file size limitations, the complete census data files are not included directly in this repository. The Census Data directory contains large datasets that should be downloaded separately. Please see the README.md file in the `/data` directory for instructions on obtaining these files.

## Key Features

- Census tract data integration from multiple sources
- Clustering and segmentation of similar geographic areas
- Interactive visualizations of geographic doppelgangers
- Statistical analysis of tract similarities

## Usage

The main notebooks and scripts demonstrate the complete workflow:
- Data extraction and preprocessing
- Segmentation analysis
- Visualization of results
- Generating geographic doppelganger recommendations

## Data Sources

- U.S. Census Bureau data (American Community Survey data)
- Additional demographic and economic datasets
- TIGER/Line Shapefiles for geographic visualization

## Requirements

Main dependencies:
- Python 3.8+
- Pandas
- NumPy
- Scikit-learn
- Matplotlib/Seaborn
- GeoPandas
- Jupyter

## License

[Specify the license under which this code is released]
