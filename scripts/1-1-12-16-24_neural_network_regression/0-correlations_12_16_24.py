import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
file_path = '/Users/test_terminal/Desktop/bgbridge/scripts/1-1-12-16-24_neural_network_regression/1__6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv'
df = pd.read_csv(file_path)

# Define independent variables (x variables)
x_vars = [
    'Age_x', 'Edu_x',
    'FAS_total_raw_x', 'FAS_total_T_x',
    'Animals_raw_x', 'Animals_T_x',
    'BNT_totalwstim_raw_x', 'BNT_totalwstim_T_x'
]

# Print summary of missing values for x_vars
print("\nMissing values summary for independent variables:")
for x in x_vars:
    missing_count = df[x].isna().sum()
    total_count = len(df)
    missing_percentage = (missing_count / total_count) * 100
    print(f"{x}: {missing_count} missing values ({missing_percentage:.2f}%)")

# Define dependent variables (y variables) - all brain regions
y_vars = [ 
    # Original cortical regions - Left hemisphere # lh.aparc.stats & rh.aparc.stats
    'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus',
    'entorhinal', 'fusiform', 'inferiorparietal', 'inferiortemporal',
    'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual',
    'medialorbitofrontal', 'middletemporal', 'parahippocampal', 'paracentral',
    'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine',
    'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
    'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal',
    'superiorparietal', 'superiortemporal', 'supramarginal', 'frontalpole',
    'temporalpole', 'transversetemporal', 'insula',
    
    # Original cortical regions - Right hemisphere
    'bankssts_rh', 'caudalanteriorcingulate_rh', 'caudalmiddlefrontal_rh', 
    'cuneus_rh', 'entorhinal_rh', 'fusiform_rh', 'inferiorparietal_rh',
    'inferiortemporal_rh', 'isthmuscingulate_rh', 'lateraloccipital_rh',
    'lateralorbitofrontal_rh', 'lingual_rh', 'medialorbitofrontal_rh',
    'middletemporal_rh', 'parahippocampal_rh', 'paracentral_rh',
    'parsopercularis_rh', 'parsorbitalis_rh', 'parstriangularis_rh',
    'pericalcarine_rh', 'postcentral_rh', 'posteriorcingulate_rh',
    'precentral_rh', 'precuneus_rh', 'rostralanteriorcingulate_rh',
    'rostralmiddlefrontal_rh', 'superiorfrontal_rh', 'superiorparietal_rh',
    'superiortemporal_rh', 'supramarginal_rh', 'frontalpole_rh',
    'temporalpole_rh', 'transversetemporal_rh', 'insula_rh',
    
    # Additional subcortical and other regions # aseg.stats
    'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
    'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate',
    'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle',
    'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
    'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel',
    'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
    'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
    'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum',
    'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area',
    'Right-VentralDC', 'Right-vessel', 'Right-choroid-plexus', '5th-Ventricle',
    'WM-hypointensities', 'Left-WM-hypointensities', 'Right-WM-hypointensities',
    'non-WM-hypointensities', 'Left-non-WM-hypointensities',
    'Right-non-WM-hypointensities', 'Optic-Chiasm', 'CC_Posterior',
    'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior'
]

# Create empty lists to store results
results = []

# Calculate correlations and p-values with proper handling of missing values
for x in x_vars:
    for y in y_vars:
        try:
            # Create a temporary DataFrame with just the two variables we're analyzing
            temp_df = df[[x, y]].copy()
            
            # Drop rows where either variable is missing (pairwise deletion)
            temp_df = temp_df.dropna()
            
            # Only calculate correlation if we have enough valid pairs
            if len(temp_df) >= 2:  # Need at least 2 pairs for correlation
                corr, p_value = stats.pearsonr(temp_df[x], temp_df[y])
                
                # Store results along with sample size
                results.append({
                    'Independent_Variable': x,
                    'Dependent_Variable': y,
                    'Correlation': corr,
                    'P_Value': p_value,
                    'Sample_Size': len(temp_df)
                })
            else:
                # Store null result if insufficient data
                results.append({
                    'Independent_Variable': x,
                    'Dependent_Variable': y,
                    'Correlation': np.nan,
                    'P_Value': np.nan,
                    'Sample_Size': len(temp_df)
                })
                
        except Exception as e:
            print(f"Error calculating correlation between {x} and {y}: {str(e)}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Create pivot tables for correlation values and p-values
correlation_matrix = results_df.pivot(
    index='Independent_Variable',
    columns='Dependent_Variable',
    values='Correlation'
)

pvalue_matrix = results_df.pivot(
    index='Independent_Variable',
    columns='Dependent_Variable',
    values='P_Value'
)

# Save detailed results with sample sizes
results_df.to_csv('0-correlation_results_detailed.csv', index=False)
correlation_matrix.to_csv('0-correlation_matrix.csv')
pvalue_matrix.to_csv('0-pvalue_matrix.csv')

# Create multiple heatmaps
chunk_size = 40
num_chunks = len(y_vars) // chunk_size + (1 if len(y_vars) % chunk_size != 0 else 0)

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(y_vars))
    
    plt.figure(figsize=(20, 6))
    sns.heatmap(correlation_matrix.iloc[:, start_idx:end_idx], 
                cmap='RdBu_r', center=0, vmin=-1, vmax=1, 
                xticklabels=True, yticklabels=True)
    plt.title(f'0-Correlation Heatmap - Part {i+1}')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'0-correlation_heatmap_part_{i+1}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Print summary statistics
print(f"\nTotal number of correlations attempted: {len(results)}")

# Calculate summary of valid correlations
valid_results = results_df.dropna(subset=['Correlation', 'P_Value'])
significant_results = valid_results[valid_results['P_Value'] < 0.05]

print(f"Number of valid correlations: {len(valid_results)}")
print(f"Number of significant correlations (p < 0.05): {len(significant_results)}")

# Sort all valid results by p-value
print("\nAll correlations sorted by p-value:")
valid_results = results_df.dropna(subset=['Correlation', 'P_Value'])
sorted_results = valid_results.sort_values('P_Value')

# Format and display all results
pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.float_format', lambda x: '{:.3e}'.format(x) if isinstance(x, float) else str(x))

print(sorted_results[['Independent_Variable', 'Dependent_Variable', 
                     'Correlation', 'P_Value', 'Sample_Size']])

# Save sorted results to CSV
sorted_results.to_csv('0-correlations_sorted_by_pvalue.csv', index=False)

# Reset display options
pd.reset_option('display.max_rows')
pd.reset_option('display.float_format')