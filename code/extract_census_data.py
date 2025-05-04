"""
A simple script to extract the gzipped census data file.
"""

import gzip
import os
import shutil

def extract_gzip_file(gzip_file_path, output_file_path=None):
    """
    Extract a gzipped file to the specified output path.
    If no output path is provided, the output file will have the same name as the input file without the .gz extension.
    
    Args:
        gzip_file_path (str): Path to the gzipped file
        output_file_path (str, optional): Path where the extracted file should be saved
    
    Returns:
        str: Path to the extracted file
    """
    # If no output path is provided, remove the .gz extension
    if output_file_path is None:
        output_file_path = gzip_file_path.rstrip('.gz')
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Extract the file
    with gzip.open(gzip_file_path, 'rb') as f_in:
        with open(output_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    print(f"File extracted successfully to: {output_file_path}")
    return output_file_path

if __name__ == "__main__":
    # File paths
    gzip_file_path = r"C:\Users\jacob\DSC680 - Applied Data Science\Project #2 (Weeks 5 thru 8)\Week #5\Census Data\SEER\us.2006_2023.tract.level.pops.txt.gz"
    output_file_path = r"C:\Users\jacob\DSC680 - Applied Data Science\Project #2 (Weeks 5 thru 8)\Week #5\SEER Data.txt"
    
    # Extract the file
    extracted_file = extract_gzip_file(gzip_file_path, output_file_path)
    
    # Display first few lines of the extracted file to verify
    try:
        with open(extracted_file, 'r', encoding='utf-8') as f:
            print("\nFirst 5 lines of the extracted file:")
            for i, line in enumerate(f):
                if i >= 5:
                    break
                print(line.strip())
    except UnicodeDecodeError:
        print("Note: The file appears to be binary or uses a different encoding.")
