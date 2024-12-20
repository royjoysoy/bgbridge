# 12-20-24 Korea Roy Seo
# MCI, Dementia, NC에서 NPTs 와 Regions of Interest volumen 평균과 std 구한 뒤, 
# 1.뇌 볼륨: 환자의 진단 그룹(MCI 또는 Dementia) 평균보다 작은 영역이 하나 이상 있음
# 2. NPT 성적: 정상군(NC) 평균에서 1 표준편차를 뺀 값보다 높은 점수를 2개 이상의 테스트에서 획득
# 이 두가지 찾아보는 scripts

# outputs:
'''
8-brain_regions_analysis_12_20_24.png
8-left_hemisphere_12_20_24.png
8-right_hemisphere_12_20_24.png	
8-decreased_regions_12_20_24.png				
8-region_npt_pairs_12_20_24.png
8-region_npt_pair_statistics_12_20_24.csv
8-group_statistics_12_20_24.csv			
8-plasticity_results_detailed_12_20_24.csv	
'''


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
            if row[region] < comparison_mean: # 해당 진단 그룹의 평균보다 작으면
                decreased_regions.append(region)
        
        # NPT 성적 체크 (NC 평균 -1SD 이상)
        preserved_tests = []
        for test in npt_columns:
            cutoff = nc_npt_stats[test]['mean'] - nc_npt_stats[test]['std']
            if row[test] >= cutoff: # NC 그룹의 (평균 - 1SD) 이상이면
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
    # 시각화 부분을 다음과 같이 수정
    # 1. 좌반구 비교
    left_regions = [r for r in input_region if not (r.endswith('_rh') or r.startswith('Right-'))]
    sem_data = []
    for group in ['NC', 'MCI', 'Dementia']:
        for region in left_regions:
            group_data = df[df['diagnosis'] == group][region]
            sem = group_data.sem()  # 표준오차 계산
            mean = group_data.mean()
            sem_data.append({
                'Group': group,
                'Region': region,
                'Mean': mean,
                'SEM': sem
            })
    
    left_sem_df = pd.DataFrame(sem_data)
    
    plt.figure(figsize=(15, 10))
    for i, group in enumerate(['NC', 'MCI', 'Dementia']):
        group_data = left_sem_df[left_sem_df['Group'] == group]
        plt.bar(np.arange(len(left_regions)) + i*0.25, group_data['Mean'],
            width=0.25, 
            yerr=group_data['SEM'],
            label=group,
            capsize=5)

    plt.xticks(np.arange(len(left_regions)) + 0.25, left_regions, rotation=45, ha='right')
    plt.legend()
    plt.title('Left Hemisphere: Group Means by Region (with SEM)', pad=20)
    plt.tight_layout()
    plt.savefig('8-left_hemisphere_12_20_24.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 우반구 비교
    right_regions = [r for r in input_region if r.endswith('_rh') or r.startswith('Right-')]
    sem_data = []
    for group in ['NC', 'MCI', 'Dementia']:
        for region in right_regions:
            group_data = df[df['diagnosis'] == group][region]
            sem = group_data.sem()  # 표준오차 계산
            mean = group_data.mean()
            sem_data.append({
                'Group': group,
                'Region': region,
                'Mean': mean,
                'SEM': sem
            })
    
    right_sem_df = pd.DataFrame(sem_data)
    
    plt.figure(figsize=(15, 10))
    for i, group in enumerate(['NC', 'MCI', 'Dementia']):
        group_data = right_sem_df[right_sem_df['Group'] == group]
        plt.bar(np.arange(len(right_regions)) + i*0.25, group_data['Mean'],
            width=0.25, 
            yerr=group_data['SEM'],
            label=group,
            capsize=5)

    plt.xticks(np.arange(len(right_regions)) + 0.25, right_regions, rotation=45, ha='right')
    plt.legend()
    plt.title('Right Hemisphere: Group Means by Region (with SEM)', pad=20)
    plt.tight_layout()
    plt.savefig('8-right_hemisphere_12_20_24.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 영역-NPT 페어링 분석만 단일 그래프로 저장
    if not plasticity_df.empty:
        plt.figure(figsize=(15, 10))
        
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
        plt.savefig('8-brain_regions_analysis_12_20_24.png', dpi=300, bbox_inches='tight')
        plt.close()
    
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