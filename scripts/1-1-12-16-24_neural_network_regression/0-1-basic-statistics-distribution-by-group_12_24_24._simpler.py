# 12-24-24 Tue Roy Seo in Korea time 
# Creates separate correlation matrices for groups: MCI, Dementia, and NC (Normal Cognition)
# Prints summary statistics for each group, including sample size and average correlation strength
# Simpler version compared to 0-1-basic-statistics-distribution-by-group_12_17_24.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# File path
file_path = '/Users/test_terminal/Desktop/bgbridge/scripts/1-1-12-16-24_neural_network_regression/1__6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv'
df = pd.read_csv(file_path)

# Define output and input variables
output_NPT = ['Age_x', 'Edu_x', 'FAS_total_T_x', 'Animals_T_x', 'BNT_totalwstim_T_x']

input_region = [
    'rostralmiddlefrontal', 'rostralmiddlefrontal_rh',  # DLPFC
    'parsopercularis', 'parsopercularis_rh',  # Inferior Frontal
    'lateralorbitofrontal', 'lateralorbitofrontal_rh',  # Lateral Orbitofrontal
    'middletemporal', 'middletemporal_rh',  # Middle Temporal
    'paracentral', 'paracentral_rh',  # Pre SMA
    'precentral', 'precentral_rh',  # Right Precentral
    'Left-Caudate', 'Right-Caudate',  # Left and Right Caudate
    'rostralanteriorcingulate', 'rostralanteriorcingulate_rh',  # ACC
    'caudalanteriorcingulate', 'caudalanteriorcingulate_rh'  # ACC
]

# Add annotations for input regions
region_annotations = {
    'rostralmiddlefrontal': 'DLPFC',
    'rostralmiddlefrontal_rh': 'DLPFC',
    'parsopercularis': 'Inferior Frontal',
    'parsopercularis_rh': 'Inferior Frontal',
    'lateralorbitofrontal': 'Lateral Orbitofrontal',
    'lateralorbitofrontal_rh': 'Lateral Orbitofrontal',
    'middletemporal': 'Middle Temporal',
    'middletemporal_rh': 'Middle Temporal',
    'paracentral': 'Pre SMA',
    'paracentral_rh': 'Pre SMA',
    'precentral': 'Right Precentral',
    'precentral_rh': 'Right Precentral',
    'Left-Caudate': 'Left and Right Caudate',
    'Right-Caudate': 'Left and Right Caudate',
    'rostralanteriorcingulate': 'ACC',
    'rostralanteriorcingulate_rh': 'ACC',
    'caudalanteriorcingulate': 'ACC',
    'caudalanteriorcingulate_rh': 'ACC',
}
annotated_regions = [f"{region} ({region_annotations[region]})" for region in input_region]

# Function to calculate correlation and p-values
def calculate_correlation_and_pvalues(data, x_vars, y_vars):
    corr_matrix = pd.DataFrame(index=x_vars, columns=y_vars)
    p_matrix = pd.DataFrame(index=x_vars, columns=y_vars)
    for x in x_vars:
        for y in y_vars:
            paired_data = data[[x, y]].dropna()
            if len(paired_data) > 1:
                corr, p_value = stats.pearsonr(paired_data[x], paired_data[y])
                corr_matrix.loc[x, y] = corr
                p_matrix.loc[x, y] = p_value
            else:
                corr_matrix.loc[x, y] = np.nan
                p_matrix.loc[x, y] = np.nan
    return corr_matrix.astype(float), p_matrix.astype(float)

# Process groups
groups = {1: 'MCI', 2: 'Dementia', 3: 'NC'}

for group, name in groups.items():
    group_df = df[df['syndrome_v2_v2_x'] == group]
    corr_matrix, p_matrix = calculate_correlation_and_pvalues(group_df, input_region, output_NPT)

    # Plot correlation heatmap with annotated y-axis
    plt.figure(figsize=(16, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={"label": "Correlation Coefficient"})
    plt.title(f"Correlation Matrix: {name}")
    plt.xticks(rotation=90)
    plt.yticks(ticks=np.arange(len(input_region)) + 0.5, labels=annotated_regions, rotation=0)  # Annotated y-axis labels
    plt.tight_layout()
    plt.savefig(f'0-1-correlation_matrix_{name}_12_24_24_annotated.png', dpi=300)
    plt.close()

    # Print summary statistics
    print(f"\n{name} Group Summary:")
    print(f"Sample size: {len(group_df)}")
    mean_corr = np.abs(corr_matrix).mean().mean()
    print(f"Average absolute correlation: {mean_corr:.3f}")
    sig_count = (p_matrix < 0.05).sum().sum()
    print(f"Significant correlations (p < 0.05): {sig_count}")
    
    # Top significant correlations
    sig_correlations = [
        (x, y, corr_matrix.loc[x, y], p_matrix.loc[x, y])
        for x in input_region for y in output_NPT
        if not np.isnan(p_matrix.loc[x, y]) and p_matrix.loc[x, y] < 0.05
    ]
    sig_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
    print("Top 5 significant correlations:")
    for i, (x, y, corr, p) in enumerate(sig_correlations[:5], 1):
        print(f"{i}. {x} - {y}: r = {corr:.3f}, p = {p:.3f}")
