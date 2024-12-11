import pandas as pd

# Read the CSV files
file1 = '3-transposed_rh_aparc_12_11_24.csv'
file2 = '2-9-1-aseg-lh-aparc-merged_rm_some_variables.csv'
output_file = '4-merged_aseg_lh_aparc-merged_rm_some_variables__transposed_rh_aparc_12_11_24.csv'

# Read the CSV files into pandas DataFrames
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Merge the DataFrames
# Left join to keep all rows from df2 and match with df1 where possible
merged_df = pd.merge(
    df2,  # left DataFrame
    df1,  # right DataFrame
    left_on='Study_ID',  # column from df2
    right_on='subject_id',  # column from df1
    how='left'  # keep all rows from df2
)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv(output_file, index=False)

# Print some information about the merge
print(f"Original number of rows in file 1: {len(df1)}")
print(f"Original number of rows in file 2: {len(df2)}")
print(f"Number of rows in merged file: {len(merged_df)}")