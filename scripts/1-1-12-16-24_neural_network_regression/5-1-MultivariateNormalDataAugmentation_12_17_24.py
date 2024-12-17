# 12-17-24 Tue In Korea time Roy Seo
# 1. Data Augmentation
# Generates syntetic samples using multivariate normal distribution
# Preserves the covariance structure between brain regions and NPT scores
# Handles missing values appropriately
# 2. Validation
# Compares original and augmented data distributions
# Checkes if correlations are preserved
# Visualizes the differences in distributions and correlations
# 3. Outputs
# Saves the augmented dataset
# Creates validation plots
# Provides summary statistics

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def augment_with_multivariate_normal(df, input_cols, output_cols, n_synthetic=100):
    """
    Augment data using multivariate normal distribution while preserving covariance structure.
    
    Parameters:
    df: Original dataframe
    input_cols: List of input column names (brain regions)
    output_cols: List of output column names (NPT scores)
    n_synthetic: Number of synthetic samples to generate
    
    Returns:
    Original and synthetic data combined
    """
    # Combine input and output columns
    all_cols = input_cols + output_cols
    
    # Handle missing values in the output columns
    df_clean = df[all_cols].copy()
    for col in output_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    # Calculate mean and covariance
    data_mean = df_clean[all_cols].mean() 
    data_cov = df_clean[all_cols].cov()
    
    # Generate synthetic samples
    synthetic_data = np.random.multivariate_normal(
        mean=data_mean,
        cov=data_cov,
        size=n_synthetic
    )
    
    # Convert to dataframe
    synthetic_df = pd.DataFrame(synthetic_data, columns=all_cols)
    
    # Combine original and synthetic data
    combined_df = pd.concat([df[all_cols], synthetic_df], ignore_index=True)
    
    return combined_df

def validate_augmentation(original_df, augmented_df, input_cols, output_cols):
    """
    Validate the augmented data by comparing distributions and correlations.
    """
    # Compare means
    original_means = original_df[input_cols + output_cols].mean()
    synthetic_means = augmented_df[input_cols + output_cols].mean()
    
    # Compare correlations
    original_corr = original_df[input_cols + output_cols].corr()
    augmented_corr = augmented_df[input_cols + output_cols].corr()
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Plot correlation differences
    diff_corr = augmented_corr - original_corr
    sns.heatmap(diff_corr, annot=True, cmap='coolwarm', center=0, ax=axes[0,0])
    axes[0,0].set_title('Correlation Difference (Augmented - Original)')
    
    # Plot mean differences
    mean_diff = (synthetic_means - original_means) / original_means * 100
    mean_diff.plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Percentage Difference in Means')
    axes[0,1].tick_params(axis='x', rotation=90)
    
    # Example distribution plots for first few variables
    for i, col in enumerate(input_cols[:2]):  # Plot first two brain regions
        sns.kdeplot(data=original_df[col], ax=axes[1,i], label='Original')
        sns.kdeplot(data=augmented_df[col], ax=axes[1,i], label='Augmented')
        axes[1,i].set_title(f'Distribution of {col}')
        axes[1,i].legend()
    
    plt.tight_layout()
    plt.savefig('5-1-augmentation_validation_12_17_24.png')
    plt.close()
    
    return mean_diff, diff_corr

# Usage example:
if __name__ == "__main__":
    # Read your data
    file_path = '/Users/test_terminal/Desktop/bgbridge/scripts/1-1-12-16-24_neural_network_regression/1__6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv'
    df = pd.read_csv(file_path)
    
    # Define input and output variables
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
    
    output_NPT = [
        'Age_x', 'Edu_x',
        'FAS_total_raw_x', 'FAS_total_T_x',
        'Animals_raw_x', 'Animals_T_x',
        'BNT_totalwstim_raw_x', 'BNT_totalwstim_T_x'
    ]
    
    # Generate synthetic data
    augmented_df = augment_with_multivariate_normal(
        df, 
        input_region, 
        output_NPT, 
        n_synthetic=386  # Generate same number as original
    )
    
    # Validate the augmentation
    mean_diff, corr_diff = validate_augmentation(
        df[input_region + output_NPT],
        augmented_df,
        input_region,
        output_NPT
    )
    
    # Save augmented data
    augmented_df.to_csv('5_1_augmented_data_multivariate_12_17_24.csv', index=False)
    
    # Print summary statistics
    print("\nMean percentage differences:")
    print(mean_diff.describe())
    print("\nCorrelation difference statistics:")
    print(corr_diff.describe())