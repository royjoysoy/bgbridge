# 12-25-24 Wed Roy Seo Korea
# To merge two files together
# file 1: 6-merged_NPT_w_o_outliers__brain_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv
# file 2: RepositorySubjectsPH-CP5192022_DATA_2022-05-19_1329.csv 오늘 다운 받음  from Carolyn's email

import pandas as pd
import numpy as np

def merge_datasets():
    # Read the datasets
    df1 = pd.read_csv('6-merged_NPT_w_o_outliers__brain_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv')
    df2 = pd.read_csv('RepositorySubjectsPH-CP5192022_DATA_2022-05-19_1329.csv')
    
    # Create a new column in df2 with just the first 8 characters of record_id
    df2['match_id'] = df2['record_id'].str[:8]
    
    # Ensure Study_ID is string type
    df1['Study_ID'] = df1['Study_ID'].astype(str)
    
    # Perform the merge
    merged_df = pd.merge(
        df1,
        df2,
        left_on='Study_ID',
        right_on='match_id',
        how='left',
        validate='1:1'  # Ensures one-to-one mapping
    )
    
    # Remove the temporary matching column
    merged_df = merged_df.drop('match_id', axis=1)
    
    # Print merge statistics
    total_rows_df1 = len(df1)
    matched_rows = merged_df['record_id'].notna().sum()
    print(f"Original rows in first dataset: {total_rows_df1}")
    print(f"Successfully matched rows: {matched_rows}")
    print(f"Unmatched rows: {total_rows_df1 - matched_rows}")
    
    # Save the merged dataset
    output_filename = '7-merged_dataset_12_25_24.csv'
    merged_df.to_csv(output_filename, index=False)
    print(f"\nMerged dataset saved as: {output_filename}")
    
    # Return info about unmatched IDs if any exist
    if total_rows_df1 - matched_rows > 0:
        unmatched_ids = df1[~df1['Study_ID'].isin(df2['record_id'].str[:8])]['Study_ID']
        print("\nUnmatched Study_IDs:")
        print(unmatched_ids.tolist())

if __name__ == "__main__":
    merge_datasets()