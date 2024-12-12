# 12-12-2024 Roy Seo
'''
All outliers in aseg.stats and lh.aparc.stats files had been previously treated, but NOT in rh.aparc.stats
I wanted to keep all stats files (aseg.stats, lh.aparc.stats, and rh.aparc.stats) in their raw form (with outliers included)
Note: Outliers were treated in NPTs but again, remained untreated in voxel counts (.stats files)
      _Dr.Kim Dae Jin suggested that outliers in .stats don't have be removed.
This script merges all variables of interest.
'''

import pandas as pd
import os

def merge_csv_files():
    # File paths
    npt_wo_outliers = "~/Desktop/bgbridge/scripts/0-2-12-09-24_dataset_merge/NPT_of_interest_w_o_outliers.csv"
    lh_aparc_file = "~/Desktop/bgbridge/scripts/0-2-12-09-24_dataset_merge/merged_lh_aparc_copy_raw_including_outliers_12_12_24.csv"
    rh_aparc_file = "~/Desktop/bgbridge/scripts/0-2-12-09-24_dataset_merge/3-transposed_rh_aparc_12_11_24.csv"
    aseg_file = "~/Desktop/bgbridge/scripts/0-2-12-09-24_dataset_merge/5-aseg_stats_raw_12_12_24.csv"
    
    output_file = "~/Desktop/bgbridge/scripts/0-2-12-09-24_dataset_merge/6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24.csv"

    # Expand user path
    npt_wo_outliers = os.path.expanduser(npt_wo_outliers)
    lh_aparc_file = os.path.expanduser(lh_aparc_file)
    rh_aparc_file = os.path.expanduser(rh_aparc_file)
    aseg_file = os.path.expanduser(aseg_file)
    output_file = os.path.expanduser(output_file)

    # Read CSV files
    df_npt = pd.read_csv(npt_wo_outliers)
    df_lh = pd.read_csv(lh_aparc_file)
    df_rh = pd.read_csv(rh_aparc_file)
    df_aseg = pd.read_csv(aseg_file)

    # Print number of rows in each file before merging
    print(f"Number of rows in NPT file (without outliers): {len(df_npt)}")
    print(f"Number of rows in LH aparc file: {len(df_lh)}")
    print(f"Number of rows in RH aparc file: {len(df_rh)}")
    print(f"Number of rows in aseg file: {len(df_aseg)}")

    # 각 파일의 Study_ID 값들을 확인
    print("\nUnique Study_IDs in each file:")
    print(f"NPT file unique IDs: {df_npt['Study_ID'].nunique()}")
    print(f"LH file unique IDs: {df_lh['Study_ID'].nunique()}")
    print(f"RH file unique IDs: {df_rh['subject_id'].nunique()}")  
    print(f"Aseg file unique IDs: {df_aseg['Subject_ID'].nunique()}")  

    # ID 형식 확인을 위한 샘플 출력
    print("\nSample IDs from each file:")
    print("NPT file:", df_npt['Study_ID'].head())
    print("LH file:", df_lh['Study_ID'].head())
    print("RH file:", df_rh['subject_id'].head())
    print("Aseg file:", df_aseg['Subject_ID'].head())

    # Study_ID 값들의 실제 형식 확인
    print("\nSample Study_ID formats:")
    for file_name, df, id_col in [("NPT", df_npt, 'Study_ID'), 
                                 ("LH", df_lh, 'Study_ID'),
                                 ("RH", df_rh, 'subject_id'),
                                 ("Aseg", df_aseg, 'Subject_ID')]:
        print(f"\n{file_name} file Sample ID representations:")
        print(df[id_col].head().apply(lambda x: f"'{x}' (type: {type(x)})"))

    # Now rename columns to match
    df_rh = df_rh.rename(columns={'subject_id': 'Study_ID'})
    df_aseg = df_aseg.rename(columns={'Subject_ID': 'Study_ID'})

    # 단계별 merge 결과 확인
    merged1 = df_npt.merge(df_lh, on='Study_ID', how='left')
    print(f"\nAfter first merge (NPT + LH): {len(merged1)} rows")

    merged2 = merged1.merge(df_rh, on='Study_ID', how='left')
    print(f"After second merge (+ RH): {len(merged2)} rows")

    final_df = merged2.merge(df_aseg, on='Study_ID', how='left')
    print(f"After final merge (+ Aseg): {len(final_df)} rows")

    # Save merged file
    final_df.to_csv(output_file, index=False)
    print(f"\nMerged file saved to: {output_file}")

if __name__ == "__main__":
    merge_csv_files()