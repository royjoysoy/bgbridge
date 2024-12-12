# 12-11-24 Roy Seo
'''
- to transpose "output_rh_aparc_copy_12_11_24.csv" file
- "3-rh_aparc_12_11_24.csv" was copied from /bgbridge/0-1-vol-ct/output_rh_aparc.csv
- After copying it, I added the regions of interest column with "_rh"
'''

import pandas as pd

def transpose_roi_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Get the column names (subject IDs) from the original file
    subject_ids = df.columns[1:].tolist()  # Skip the first column
    
    # Get the ROI names from the first column, starting from row 2
    roi_names = df.iloc[0:34, 0].tolist()
    
    # Get the data values, excluding the header row and first column
    data = df.iloc[0:34, 1:].values
    
    # Transpose the data
    data_transposed = data.T
    
    # Create a new DataFrame with the transposed data
    df_transposed = pd.DataFrame(data_transposed, columns=roi_names)
    
    # Add the subject IDs as the first column
    df_transposed.insert(0, 'subject_id', subject_ids)
    
    return df_transposed

# Example usage
if __name__ == "__main__":
    # Replace with your input file path
    input_file = "~/Desktop/bgbridge/scripts/0-2-120924_dataset_merge/3-rh_aparc_12_11_24.csv"
    
    # Transpose the data
    result = transpose_roi_data(input_file)
    
    # Save to a new CSV file
    output_file = "~/Desktop/bgbridge/scripts/0-2-120924_dataset_merge/3-transposed_rh_aparc_12_11_24.csv"
    result.to_csv(output_file, index=False)
    print(f"Transposed data saved to {output_file}")