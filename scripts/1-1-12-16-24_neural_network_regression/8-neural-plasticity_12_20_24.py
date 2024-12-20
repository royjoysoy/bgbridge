import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_brain_regions(data_path):
    # 데이터 로드
    df = pd.read_csv(data_path)
    
    # 관심 영역 정의
    input_region = [
        'rostralmiddlefrontal', 'caudalmiddlefrontal', 'parsopercularis', 
        'parsorbitalis', 'parstriangularis', 'lateralorbitofrontal', 
        'middletemporal', 'superiorfrontal', 'precentral', 'Left-Caudate',
        'rostralanteriorcingulate', 'caudalanteriorcingulate',
        'rostralmiddlefrontal_rh', 'caudalmiddlefrontal_rh', 'parsopercularis_rh',
        'parsorbitalis_rh', 'parstriangularis_rh', 'lateralorbitofrontal_rh',
        'middletemporal_rh', 'superiorfrontal_rh', 'precentral_rh', 'Right-Caudate',
        'rostralanteriorcingulate_rh', 'caudalanteriorcingulate_rh'
    ]
    
    # NPT 컬럼
    npt_columns = ['FAS_total_T_x', 'Animals_T_x', 'BNT_totalwstim_T_x']
    
    # 진단 그룹 매핑
    diagnosis_map = {1: 'MCI', 2: 'Dementia', 3: 'NC'}
    df['diagnosis'] = df['syndrome_v2_v2_x'].map(diagnosis_map)
    
    # 각 그룹별, 각 영역별 평균과 표준편차 계산
    group_stats = {}
    for group in ['NC', 'MCI', 'Dementia']:
        group_stats[group] = {}
        group_data = df[df['diagnosis'] == group]
        for region in input_region:
            mean = group_data[region].mean()
            std = group_data[region].std()
            group_stats[group][region] = {'mean': mean, 'std': std}
    
    # 결과를 저장할 DataFrame 생성
    stats_data = []
    for group in group_stats:
        for region in group_stats[group]:
            stats_data.append({
                'Group': group,
                'Region': region,
                'Mean': group_stats[group][region]['mean'],
                'SD': group_stats[group][region]['std']
            })
    stats_df = pd.DataFrame(stats_data)
    
    # NC 그룹의 NPT 통계 계산
    nc_group = df[df['diagnosis'] == 'NC']
    nc_npt_stats = {}
    for col in npt_columns:
        mean = nc_group[col].mean()
        std = nc_group[col].std()
        nc_npt_stats[col] = {'mean': mean, 'std': std}
    
    # 가소성 판단 함수
    def has_plasticity(row):
        # 뇌 볼륨 감소 체크 (MCI/Dementia 그룹 평균보다 작은지)
        decreased_regions = []
        patient_group = row['diagnosis']
        
        for region in input_region:
            comparison_mean = group_stats[patient_group][region]['mean']
            if row[region] < comparison_mean:
                decreased_regions.append(region)
        
        # NPT 성적 체크 (NC 평균 -1SD 이상)
        preserved_tests = []
        for test in npt_columns:
            cutoff = nc_npt_stats[test]['mean'] - nc_npt_stats[test]['std']
            if row[test] >= cutoff:
                preserved_tests.append(test)
        
        return (len(decreased_regions) >= 1 and len(preserved_tests) >= 2), decreased_regions, preserved_tests
    
    # 가소성 분석
    plasticity_results = []
    for idx, row in df.iterrows():
        if row['diagnosis'] in ['MCI', 'Dementia']:
            has_plast, dec_regions, pres_tests = has_plasticity(row)
            if has_plast:
                plasticity_results.append({
                    'ID': row['Study_ID'],
                    'Diagnosis': row['diagnosis'],
                    'Decreased_Regions': dec_regions,
                    'Preserved_Tests': pres_tests,
                    'Age': row['Age_x']
                })
    
    plasticity_df = pd.DataFrame(plasticity_results)
    
    # 시각화 - 여러 개의 그림으로 분할
    # 1. 좌반구 비교
    fig1 = plt.figure(figsize=(15, 10))
    left_regions = [r for r in input_region if not (r.endswith('_rh') or r.startswith('Right-'))]
    left_data = stats_df[stats_df['Region'].isin(left_regions)]
    sns.barplot(data=left_data, x='Region', y='Mean', hue='Group', 
                capsize=0.2, errwidth=2, alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.title('Left Hemisphere: Group Means by Region (with SE)', pad=20)
    plt.tight_layout()
    plt.savefig('8-left_hemisphere_12_20_24.png', dpi=300, bbox_inches='tight')
    
    # 2. 우반구 비교
    fig2 = plt.figure(figsize=(15, 10))
    right_regions = [r for r in input_region if r.endswith('_rh') or r.startswith('Right-')]
    right_data = stats_df[stats_df['Region'].isin(right_regions)]
    sns.barplot(data=right_data, x='Region', y='Mean', hue='Group',
                capsize=0.2, errwidth=2, alpha=0.8)
    plt.xticks(rotation=45, ha='right')
    plt.title('Right Hemisphere: Group Means by Region (with SE)', pad=20)
    plt.tight_layout()
    plt.savefig('8-right_hemisphere_12_20_24.png', dpi=300, bbox_inches='tight')
    
    # 3. 감소된 영역 분포
    fig3 = plt.figure(figsize=(12, 8))
    if not plasticity_df.empty:
        region_counts = {}
        for regions in plasticity_df['Decreased_Regions']:
            for region in regions:
                region_counts[region] = region_counts.get(region, 0) + 1
        
        region_df = pd.DataFrame(list(region_counts.items()), 
                               columns=['Region', 'Count'])
        region_df = region_df.sort_values('Count', ascending=True)
        
        sns.barplot(data=region_df, y='Region', x='Count', palette="YlOrRd")
        plt.title('Frequency of Decreased Regions in Plasticity Cases', pad=20)
        plt.tight_layout()
        plt.savefig('8-decreased_regions_12_20_24.png', dpi=300, bbox_inches='tight')
    
    # 4. 영역-NPT 페어링 분석
    if not plasticity_df.empty:
        fig4 = plt.figure(figsize=(15, 10))
        
        # 각 영역-NPT 페어의 빈도 계산
        pair_counts = {}
        for _, row in plasticity_df.iterrows():
            for region in row['Decreased_Regions']:
                for test in row['Preserved_Tests']:
                    pair = (region, test)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        # 상위 20개 페어 추출
        pair_df = pd.DataFrame([
            {'Region-NPT Pair': f"{region} & {test}", 'Count': count}
            for (region, test), count in pair_counts.items()
        ])
        pair_df = pair_df.sort_values('Count', ascending=True).tail(20)
        
        sns.barplot(data=pair_df, y='Region-NPT Pair', x='Count', palette="viridis")
        plt.title('Top 20 Region-NPT Pairs Showing Plasticity', pad=20)
        plt.tight_layout()
        plt.savefig('8-region_npt_pairs_12_20_24.png', dpi=300, bbox_inches='tight')
    
    # 5. 상세 통계 정보를 CSV에 추가
    pair_stats = pd.DataFrame([
        {'Region-NPT Pair': f"{region} & {test}", 
         'Count': count,
         'Percentage': (count/len(plasticity_df))*100}
        for (region, test), count in pair_counts.items()
    ]).sort_values('Count', ascending=False)
    
    pair_stats.to_csv('8-region_npt_pair_statistics_12_20_24.csv', index=False)
    
    return plasticity_df, [fig1, fig2, fig3, fig4], stats_df, pair_stats

# 실행
if __name__ == "__main__":
    file_path = "1__6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv"
    results, figs, stats, pair_stats = analyze_brain_regions(file_path)
    
    print("\nVisualization has been saved as separate files:")
    print("1. '8-left_hemisphere_12_20_24.png'")
    print("2. '8-right_hemisphere_12_20_24.png'")
    print("3. '8-decreased_regions_12_20_24.png'")
    print("4. '8-region_npt_pairs_12_20_24.png'")
    print("\nDetailed statistics have been saved as:")
    print("1. '8-region_npt_pair_statistics_12_20_24.csv'")
    
    # Top 10 Region-NPT pairs 출력
    print("\nTop 10 Region-NPT pairs showing plasticity:")
    print(pair_stats.head(10)[['Region-NPT Pair', 'Count', 'Percentage']].to_string(index=False))