# 12-12-2024 Roy Seo
# outliers를 제거하고 대체 하기전 raw file을 찾을 수 없어서 다시 만듦

import pandas as pd
import os

# 경로 설정
listsubj = os.path.expanduser('~/fsgroupAug11/subj')
listsubjtoextract = [f for f in os.listdir(listsubj) if not f.startswith('.')]

# 첫 번째 피험자의 파일에서 ROI 이름 가져오기 (헤더용)
first_subj = listsubjtoextract[0]
first_file = os.path.join(listsubj, first_subj, 'stats/aseg.stats')
with open(first_file, 'r') as f:
    lines = f.readlines()[80:125]  # 81번째 줄부터 125번째 줄까지
    roi_names = [line.split()[4] for line in lines if line.strip()]  # 5번째 컬럼의 ROI 이름

# 결과를 저장할 딕셔너리
data_dict = {'Subject_ID': []}

# 각 ROI에 대한 빈 리스트 초기화
for roi in roi_names:
    data_dict[roi] = []

# 각 피험자의 데이터 읽기
for subj in listsubjtoextract:
    placevolume = os.path.join(listsubj, subj, 'stats/aseg.stats')
    try:
        # 파일 읽기
        with open(placevolume, 'r') as f:
            lines = f.readlines()[80:125]
        
        # 피험자 ID 추가 ("sub-" 제거)
        data_dict['Subject_ID'].append(subj.replace('sub-', ''))
        
        # 4번째 컬럼(볼륨) 추출
        for i, line in enumerate(lines):
            if line.strip():
                columns = line.split()
                if len(columns) >= 4:
                    data_dict[roi_names[i]].append(float(columns[3]))
                    
    except Exception as e:
        print(f"Error reading data for subject {subj}: {e}")

# DataFrame 생성 및 transpose
results = pd.DataFrame(data_dict)
results_t = results.set_index('Subject_ID').T  # transpose하되 Subject_ID는 컬럼으로 유지

# 결과 저장
output_path = os.path.expanduser('~/fsgroupAug11/bgbridge/scripts/0-2-12-09-24_dataset_merge/5-aseg_stats_raw.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
results.to_csv(output_path, index=False)

# 결과 확인
print(results)
