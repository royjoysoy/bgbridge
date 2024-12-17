# 12-17-24 Tue Roy Seo in Korea time 
# creates separate correlation matrices for each group 1: MCI, 2: Dementia, and 3 Subjective Memory Complaint / Normal Cognition if there is 4: Unknown -Defer to Record
# prints summary statistics for each groupm including sample size and average correlation strength

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the data (using your file path)
file_path = '/Users/test_terminal/Desktop/bgbridge/scripts/1-1-12-16-24_neural_network_regression/1__6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv'
df = pd.read_csv(file_path)

# Define your variables
output_NPT = [
    'Age_x', 'Edu_x',
    'FAS_total_raw_x', 'FAS_total_T_x',
    'Animals_raw_x', 'Animals_T_x',
    'BNT_totalwstim_raw_x', 'BNT_totalwstim_T_x'
]

input_region = [ 
    'rostralmiddlefrontal', 'rostralmiddlefrontal_rh',
    'caudalmiddlefrontal', 'caudalmiddlefrontal_rh',
    'parsopercularis', 'parsopercularis_rh',
    'parsorbitalis', 'parsorbitalis_rh',
    'parstriangularis', 'parstriangularis_rh',
    'lateralorbitofrontal', 'lateralorbitofrontal_rh',
    'middletemporal', 'middletemporal_rh',
    'superiorfrontal', 'superiorfrontal_rh',
    'precentral', 'precentral_rh',
    'Left-Caudate', 'Right-Caudate',
    'rostralanteriorcingulate', 'rostralanteriorcingulate_rh',
    'caudalanteriorcingulate', 'caudalanteriorcingulate_rh'
]

# Function to calculate correlation and p-values
def calculate_correlation_and_pvalues(data, x_vars, y_vars):
    corr_matrix = pd.DataFrame(np.zeros((len(x_vars), len(y_vars))), index=x_vars, columns=y_vars)
    p_matrix = pd.DataFrame(np.zeros((len(x_vars), len(y_vars))), index=x_vars, columns=y_vars)
    
    for x in x_vars:
        for y in y_vars:
            # Get data for both variables and drop rows where either has NaN
            paired_data = data[[x, y]].dropna()
            
            if len(paired_data) > 1:  # Check if we have enough data points
                corr, p_value = stats.pearsonr(paired_data[x], paired_data[y])
                corr_matrix.loc[x, y] = corr
                p_matrix.loc[x, y] = p_value
            else:
                corr_matrix.loc[x, y] = np.nan
                p_matrix.loc[x, y] = np.nan
            
    return corr_matrix, p_matrix

# Create correlation matrices and p-value matrices for each group
groups = [1, 2, 3]
group_names = {1: 'MCI', 2: 'Dementia', 3: 'NC'}

for group in groups:
    # Filter data for current group
    group_df = df[df['syndrome_v2_v2_x'] == group]
    
    # Calculate correlations and p-values
    corr_matrix, p_matrix = calculate_correlation_and_pvalues(group_df, input_region, output_NPT)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
    
    # Plot correlation heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'}, ax=ax1)
    ax1.set_title(f'Correlation Matrix: {group_names[group]}')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
    
    # Plot p-value heatmap
    # Create a mask for significant p-values
    sig_mask = p_matrix < 0.05
    
    # Plot p-values with significance markers
    sns.heatmap(p_matrix, annot=True, cmap='YlOrRd_r', fmt='.3f',
                cbar_kws={'label': 'P-value'}, ax=ax2)
    
    # Add stars for significant correlations
    for i in range(len(input_region)):
        for j in range(len(output_NPT)):
            if sig_mask.iloc[i, j]:
                ax2.text(j + 0.5, i + 0.5, '*', 
                        ha='center', va='center', color='white',
                        fontweight='bold', fontsize=12)
    
    ax2.set_title(f'P-values Matrix: {group_names[group]}\n* indicates p < 0.05')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'0-1_LanConbrain_npt_cross_correlations_with_pvalues_group_{group}_12_17_24.png', 
                bbox_inches='tight', dpi=300)
    plt.close()

# Print summary statistics for each group
for group in groups:
    group_df = df[df['syndrome_v2_v2_x'] == group]
    corr_matrix, p_matrix = calculate_correlation_and_pvalues(group_df, input_region, output_NPT)
    
    print(f"\nGroup {group_names[group]} Summary:")
    print(f"Number of subjects: {len(group_df)}")
    
    # Calculate average correlation excluding NaN values
    mean_corr = np.abs(corr_matrix).mean().mean()
    if not np.isnan(mean_corr):
        print(f"Average absolute correlation: {mean_corr:.3f}")
    else:
        print("Average absolute correlation: No valid correlations")
    
    # Count significant correlations
    sig_count = (p_matrix < 0.05).sum().sum()
    print(f"Number of significant correlations (p < 0.05): {sig_count}")
    
    # Find strongest significant correlations
    sig_correlations = []
    for x in input_region:
        for y in output_NPT:
            if not np.isnan(p_matrix.loc[x, y]) and p_matrix.loc[x, y] < 0.05:
                sig_correlations.append((x, y, corr_matrix.loc[x, y], p_matrix.loc[x, y]))
    
    if sig_correlations:
        print("\nTop 5 strongest significant correlations:")
        sig_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        for i, (region, npt, corr, p_val) in enumerate(sig_correlations[:5], 1):
            print(f"{i}. {region} - {npt}: r = {corr:.3f}, p = {p_val:.3f}")