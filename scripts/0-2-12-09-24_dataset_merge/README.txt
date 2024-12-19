# 12-19-2024 Thur In Korea Time Roy Seoy
# README file for this folder 0-2-12-09-24_dataset merge


# 주로 등장하는 용어
NPT: Neuropyschological Tests (이 실험에서는 언어능력에 관련된 NPTs들만 골랐다.)
aseg.stats: Automated Segmentation의 약자로 FreeSurfer가 뇌의 피질하 (subcortical) 영역과 몇몇 주요 구조를 분할한 결과
            (자동으로 분할된 피질 및 피질하 구조의 '부피' 단위 mm^3 세제곱 밀리미터 를 제공)
            나는 4번째 칼럼 Volume_mm3 (mm3)를 사용함
lh (or rh).aparc.stats: Automated Parcellation의 약자로, 피질을 세분화(Parcellation)하여 뇌 영역별 수치를 계산한 결과
                피질 영역은 Desikan-Killiany atlas 또는 다른 atlas에 따라 나뉨
                나는 4 번째 column 즉 grayvol을 사용함
               




----
1-12-09-24_missing_data.py
To find out NPTs which has less than 50, 100, 150 missing datasets

----
2-12-09-24_test_nn.py:
neural network test로 돌려 본 것입니다. 매우 basic 한 model architecture

----
2-9-1-aseg-lh-aparc-merged_rm_some_variables.csv
필요없는 variables 을 지웠습니다. 주로 text로 대답한 (숫자 values가 아닌) 것들, analyses에 사용되지 않을 것들을 manually 지웠습니다. 

----
3-rh_aparc_12_11_24.csv
bgbrdige/0-1-vol-ct/output_rh_aparc.csv를 copy함 
regions of interest에 _rh라는 구분이 안되어 있어서 manually 이름에다가 "_rh" 라는 것을 썼습니다. 

----
3-transpose_rh_aparc_12_11_24.py
위의 3-rh_aparc_12_11_24.csv 를 transpose하기위한 python script

----
3-transposed_rh_aparc_12_11_24.csv
위의 3-rh_aparc_12_11_24.csv 를 transpose하고 제대로 값이 transpose되었는지 manually 몇 개 골라 확인함
----
4-merge_transposed_rh_aparc_12_11_24__2-9-1_aseg_lh_aparc_merged_rm_some_variables.py
3-transposed_rh_aparc_12_11_24.csv 와  2-9-1-aseg-lh-aparc-merged_rm_some_variables.csv를 merge하기 위한 python script

----
4-merged_aseg_lh_aparc-merged_rm_some_variables__transposed_rh_aparc_12_11_24.csv
4-merge_transposed_rh_aparc_12_11_24__2-9-1_aseg_lh_aparc_merged_rm_some_variables.py의 결과물

----
5-aseg_stats_raw_12_12_24.csv
5-aseg_stats_raw_12_12_24.py의 결과물

----
5-aseg_stats_raw_12_12_24.py
aseg.stats 의 outliers를 제거하고 대체 하기전 raw file을 찾을 수 없어서 다시 긁어옴
transposed 되어서 merge된 3-transposed_rh_aparc_12_11_24.csv는 outlier가 처리가 안되어있는데
그 외의 모든 것들은 outliers 가 처리 되어있었음. 그래서 Kim Dae Jin박사님이 volume은 outlier처리를 잘 안한다고 조언 해준것을 바탕으로 
NPTs는 outlier 처리, aseg.stats and lh_aparc.stats, rh_aparc_stats은 outliers 처리 안된 것 사용
outlier 처리는 bgbridge/2-4-outliersbye-lh_aparc.ipynb 참조 3 SD higher or lower than mean을 mean으로 대체)  
그대로 나두고 volume은 raw (즉 outliers 처리되기전) 파일을 쓰려고 함


----
6-merged_NPT_w_o_outliers__brain_w_outliers_12_12_24.csv
6-merged_NPT_w_o_outliers__braub_w_outliers_12_12_24.py의 결과물

----
6-merged_NPT_w_o_outliers__brain_w_outliers_12_12_24.py
NPTs는 outliers가 처리 되었다. 그래서 without "w_o_outliers" 이라고 썼고, 
volume은 outliers가 있는 raw data 여서 with (w_outliers)라고 
파일 이름을 정하였다.

----
6-merged_NPT_w_o_outliers__brain_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv
이상하게 같은 subjects가 여러 rows에 걸쳐 반복되어있는 현상이 있었다. 그래서 반복되는 rows들을 지웠다. script없이 직접 지움

----
NPT_copy.csv
NPT_of_interest_w_o_outliers.csv
dx-syndrome_copy.csv
fsgroupAug11_subj_4-1-add-dx-lh_aparc_modified_for_aseg_stats_copy.csv
handedness_copy.csv
merged_lh_aparc_copy_raw_including_outliers_12_12_24.csv

이 파일들은 다른 폴더에서 카피해 온 파일들이다. 
