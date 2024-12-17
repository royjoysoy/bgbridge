# 12-17-2024 in Korea time Roy Seo
# To calculate basic statistics (mean, std, skewness, missing values, min, max) for all variables
# all variables (Age, Edu, NPTs and volumes of some language control regions)
# To create distribution plots with histograms and kernel density estimation
# To generate correlation heatmaps to show relationships between variables
# Save all reaults as CSV files and plots as PNG files

# All of these is to determine which augmentation methods are the best for my dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the data
file_path = '/Users/test_terminal/Desktop/bgbridge/scripts/1-1-12-16-24_neural_network_regression/1__6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv'
df = pd.read_csv(file_path)

# Define input and output variables (from your script)
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

# Function to calculate statistics
def get_statistics(data, columns):
    stats_dict = {}
    for col in columns:
        stats_dict[col] = {
            'mean': data[col].mean(),
            'std': data[col].std(),
            'skewness': data[col].skew(),
            'missing_values': data[col].isna().sum(),
            'min': data[col].min(),
            'max': data[col].max()
        }
    return pd.DataFrame(stats_dict).T

# Calculate statistics for both sets of variables
region_stats = get_statistics(df, input_region)
npt_stats = get_statistics(df, output_NPT)

# Print statistics
print("\nBrain Region Statistics:")
print(region_stats)
print("\nNPT Statistics:")
print(npt_stats)

# Save statistics to CSV
region_stats.to_csv('0-LanCon_brain_region_statistics_12_17_24.csv')
npt_stats.to_csv('0-npt_statistics_12_17_24.csv')

# Create distribution plots
def plot_distributions(data, variables, title, filename, ncols=3):
    nrows = (len(variables) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5*nrows))
    axes = axes.flatten()

    for idx, var in enumerate(variables):
        # Histogram with kernel density estimation
        sns.histplot(data=data, x=var, kde=True, ax=axes[idx])
        axes[idx].set_title(f'{var}\nSkewness: {data[var].skew():.2f}')
        axes[idx].tick_params(axis='x', rotation=45)

    # Remove empty subplots
    for idx in range(len(variables), len(axes)):
        fig.delaxes(axes[idx])

    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Create plots
plot_distributions(df, input_region, 'Brain Region Volume Distributions', '0-LanCon_brain_region_distributions_12_17_24.png')
plot_distributions(df, output_NPT, 'NPT Score Distributions', '0-npt_distributions_12_17_24.png')

# Create correlation heatmaps
def plot_correlation_heatmap(data, variables, title, filename):
    plt.figure(figsize=(12, 10))
    correlation_matrix = data[variables].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

# Create correlation heatmaps
plot_correlation_heatmap(df, input_region, 'Brain Region Volume Correlations', '0-LanConbrain_region_correlations_12_17_24.png')
plot_correlation_heatmap(df, output_NPT, 'NPT Score Correlations', '0-npt_correlations_12_17_24.png')

# Optional: Cross-correlation between brain regions and NPT scores
plt.figure(figsize=(15, 8))
cross_correlation = df[input_region + output_NPT].corr().loc[input_region, output_NPT]
sns.heatmap(cross_correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Brain Region - NPT Score Correlations')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('0-LanConbrain_npt_cross_correlations_12_17_24.png', bbox_inches='tight', dpi=300)
plt.close()