# 12-09-2024 Roy Seo
# Usage: to find out NPTs which has less than 50, 100, 150 missing datasets

import pandas as pd
import numpy as np
import os

def analyze_npt_headers(file_path):
    df = pd.read_csv(file_path)
    
    # Count total columns
    num_columns = len(df.columns)
    print(f"Total number of columns: {num_columns}")
    
    relevant_cols = df.columns[0:369]
    headers_over_350 = []
    headers_over_300 = []
    headers_over_250 = []
    
    for col in relevant_cols:
        total_values = len(df[col])
        missing_values = df[col].isna().sum()
        valid_values = total_values - missing_values
        
        if valid_values > 350 and missing_values <= 50:
            headers_over_350.append(col)
            print(f"{col}: {valid_values} valid values, {missing_values} missing")

        if valid_values > 350 and  51 <= missing_values <= 100:
            headers_over_300.append(col)
            print(f"{col}: {valid_values} valid values, {missing_values} missing")
        
        if valid_values > 350 and  101 <= missing_values <= 100:
            headers_over_250.append(col)
            print(f"{col}: {valid_values} valid values, {missing_values} missing")
    
    return headers_over_350, headers_over_300, headers_over_250

def analyze_missing_data(file_path):
    df = pd.read_csv(file_path)
    relevant_cols = df.columns[3:369]
    
    print("\nMissing data count per column:")
    print("=" * 50)
    
    for col in relevant_cols:
        missing_values = df[col].isna().sum()
        total_values = len(df[col])
        print(f"{col}: {missing_values}/{total_values} missing")

# Execute analysis
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'NPT_copy.csv')

headers = analyze_npt_headers(file_path)
print("=" * 50)
analyze_missing_data(file_path)