import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx  # Added import for networkx

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

def plot_bar_charts(corr_matrix, group_name):
    """Plot bar charts for each cognitive test."""
    n_tests = len(output_NPT)
    fig, axes = plt.subplots(n_tests, 1, figsize=(12, 4*n_tests))
    
    for idx, test in enumerate(output_NPT):
        data = corr_matrix[test].sort_values()
        colors = ['red' if x >= 0 else 'blue' for x in data]
        
        axes[idx].barh(range(len(data)), data, color=colors)
        axes[idx].set_yticks(range(len(data)))
        axes[idx].set_yticklabels([f"{region} ({region_annotations[region]})" for region in data.index])
        axes[idx].set_title(f'{test} Correlations')
        axes[idx].axvline(x=0, color='black', linestyle='-', alpha=0.2)
        
    plt.tight_layout()
    plt.savefig(f'correlation_bars_{group_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_radar_chart(corr_matrix, group_name):
    """Plot radar chart for correlations."""
    # Prepare data for radar chart
    fig = go.Figure()
    
    for test in output_NPT:
        fig.add_trace(go.Scatterpolar(
            r=corr_matrix[test],
            theta=[f"{region}\n({region_annotations[region]})" for region in corr_matrix.index],
            name=test,
            fill='toself'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-0.4, 0.4]
            )),
        showlegend=True,
        title=f"Radar Chart of Correlations - {group_name}"
    )
    
    fig.write_html(f'correlation_radar_{group_name}.html')

def plot_network_graph(corr_matrix, p_matrix, group_name, p_threshold=0.05):
    """Plot network graph of significant correlations."""
    G = nx.Graph()
    
    # Add nodes
    for region in input_region:
        G.add_node(region, bipartite=0)
    for test in output_NPT:
        G.add_node(test, bipartite=1)
    
    # Add edges for significant correlations
    for region in input_region:
        for test in output_NPT:
            if p_matrix.loc[region, test] < p_threshold:
                correlation = corr_matrix.loc[region, test]
                if abs(correlation) > 0.2:  # Only show stronger correlations
                    G.add_edge(region, test, weight=abs(correlation),
                             correlation=correlation)
    
    # Create layout
    pos = nx.spring_layout(G)
    
    # Plot
    plt.figure(figsize=(15, 15))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=input_region, node_color='lightblue',
                          node_size=2000, alpha=0.7)
    nx.draw_networkx_nodes(G, pos, nodelist=output_NPT, node_color='lightgreen',
                          node_size=2000, alpha=0.7)
    
    # Draw edges with different colors based on correlation
    edges = G.edges()
    weights = [G[u][v]['correlation'] for u, v in edges]
    nx.draw_networkx_edges(G, pos, edge_color=weights, edge_cmap=plt.cm.RdBu,
                          width=2, edge_vmin=-0.4, edge_vmax=0.4)
    
    # Add labels
    nx.draw_networkx_labels(G, pos)
    
    plt.title(f"Network of Significant Correlations - {group_name}")
    plt.axis('off')
    plt.savefig(f'correlation_network_{group_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Process groups
groups = {1: 'MCI', 2: 'Dementia', 3: 'NC'}

for group, name in groups.items():
    group_df = df[df['syndrome_v2_v2_x'] == group]
    corr_matrix, p_matrix = calculate_correlation_and_pvalues(group_df, input_region, output_NPT)
    
    # Generate all visualizations
    plot_bar_charts(corr_matrix, name)
    plot_radar_chart(corr_matrix, name)
    plot_network_graph(corr_matrix, p_matrix, name)
    
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
    print("\nTop 5 significant correlations:")
    for i, (x, y, corr, p) in enumerate(sig_correlations[:5], 1):
        print(f"{i}. {x} ({region_annotations[x]}) - {y}: r = {corr:.3f}, p = {p:.3f}")