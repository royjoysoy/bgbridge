# 12-16-24 Mon (In Korea time) Roy Seo
# 1-1-12-16-24_neural_network_regression

---------------------------  12-16-24 Mon  ----------------------------------

1.1-1-12-16-24_neural_network_regression folder 만듦

2. from 0-2-12-09-24_dataset_merge folder에서 
6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv 파일
복사해옴
파일의 새이름
1__6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv

2. dataset variables 
총 386개의 datasets
variables: 
Age
Edu
ethnic backgroud 1
ethnic backgroud 2

FAS_total_raw
FAS_total_T

Animal_raw
Animal_T

BNT_total
BNT_raw

Handedness 모두 1

- lh & rh (34 * 2 = 68 variables) - lh.aparc.stats & rh.aparc.stats
bankssts	
caudalanteriorcingulate	
caudalmiddlefrontal	
cuneus	
entorhinal	
fusiform	
inferiorparietal	
inferiortemporal	
isthmuscingulate	
lateraloccipital	
lateralorbitofrontal	
lingual	
medialorbitofrontal	
middletemporal	
parahippocampal	
paracentral	
parsopercularis	
parsorbitalis	
parstriangularis	
pericalcarine	
postcentral	
posteriorcingulate	
precentral	
precuneus	
rostralanteriorcingulate	
rostralmiddlefrontal	
superiorfrontal	
superiorparietal	
superiortemporal	
supramarginal	
frontalpole	
temporalpole	
transversetemporal	
insula

bankssts	
caudalanteriorcingulate	
caudalmiddlefrontal	
cuneus	
entorhinal	
fusiform	
inferiorparietal	
inferiortemporal	
isthmuscingulate	
lateraloccipital	
lateralorbitofrontal	
lingual	
medialorbitofrontal	
middletemporal	
parahippocampal	
paracentral	
parsopercularis	
parsorbitalis	
parstriangularis	
pericalcarine	
postcentral	
posteriorcingulate	
precentral	
precuneus	
rostralanteriorcingulate	
rostralmiddlefrontal	
superiorfrontal	
superiorparietal	
superiortemporal	
supramarginal	
frontalpole	
temporalpole	
transversetemporal	
insula_rh

- (45 variables - aseg.stats
Left-Lateral-Ventricle	
Left-Inf-Lat-Vent	
Left-Cerebellum-White-Matter	
Left-Cerebellum-Cortex	
Left-Thalamus-Proper	
Left-Caudate	
Left-Putamen	
Left-Pallidum	
3rd-Ventricle	
4th-Ventricle	
Brain-Stem	
Left-Hippocampus	
Left-Amygdala	
CSF	
Left-Accumbens-area	
Left-VentralDC	
Left-vessel	
Left-choroid-plexus	
Right-Lateral-Ventricle	
Right-Inf-Lat-Vent	
Right-Cerebellum-White-Matter	
Right-Cerebellum-Cortex	
Right-Thalamus-Proper	
Right-Caudate	
Right-Putamen	
Right-Pallidum	
Right-Hippocampus	
Right-Amygdala	
Right-Accumbens-area	
Right-VentralDC	
Right-vessel	
Right-choroid-plexus	
5th-Ventricle	
WM-hypointensities	
Left-WM-hypointensities	
Right-WM-hypointensities	
non-WM-hypointensities	
Left-non-WM-hypointensities	
Right-non-WM-hypointensities	
Optic-Chiasm	
CC_Posterior	
CC_Mid_Posterior	
CC_Central	
CC_Mid_Anterior	
CC_Anterior

3. 0-correlations_12_16_24.py로 대충 correlation coefficients와 p-values 봄
- multiple comparison correction은 안했음

4. 1-nn_regresion12_16_24.py 매우 basic 한 모델로 neural network regression봄
'''  
  -- Results 4 --
Training Loss: [0.6980955600738525, 0.6238742470741272]
Test Loss: [1.0110758543014526, 0.7723933458328247]
3/3 [==============================] - 0s 1ms/step
R-squared for Age_x: 0.3287
R-squared for Edu_x: 0.0377
R-squared for FAS_total_raw_x: 0.1041
R-squared for FAS_total_T_x: 0.1047
R-squared for Animals_raw_x: 0.0262
R-squared for Animals_T_x: 0.1599
R-squared for BNT_totalwstim_raw_x: 0.2597
R-squared for BNT_totalwstim_T_x: 0.1327

Model Summary:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 128)               14592     
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 64)                8256      
                                                                 
 dropout_1 (Dropout)         (None, 64)                0         
                                                                 
 dense_2 (Dense)             (None, 32)                2080      
                                                                 
 dense_3 (Dense)             (None, 8)                 264       
                                                                 
=================================================================
Total params: 25,192
Trainable params: 25,192
Non-trainable params: 0
_________________________________________________________________
'''
5. 
4 번 모델은 StandardScaler로 했고 5번은 MinMax Scaler로 했는데 StandardScaler가 성능이 훨신 좋군요
'''
  -- Results 5 --
R-squared for Age_x: 0.0051
R-squared for Edu_x: 0.0755
R-squared for FAS_total_raw_x: 0.1052
R-squared for FAS_total_T_x: 0.1932
R-squared for Animals_raw_x: 0.0032
R-squared for Animals_T_x: 0.1433
R-squared for BNT_totalwstim_raw_x: 0.2327
R-squared for BNT_totalwstim_T_x: 0.1746

Model Summary:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 128)               640       
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 64)                8256      
                                                                 
 dropout_1 (Dropout)         (None, 64)                0         
                                                                 
 dense_2 (Dense)             (None, 32)                2080      
                                                                 
 dense_3 (Dense)             (None, 8)                 264       
                                                                 
=================================================================
Total params: 11,240
Trainable params: 11,240
Non-trainable params: 0
_________________________________________________________________

'''

6. language control only regions
'''
  -- Results 6 --
Training Loss: [0.890381932258606, 0.7179588675498962]
Test Loss: [1.0340725183486938, 0.7910915613174438]
3/3 [==============================] - 0s 1ms/step
R-squared for Age_x: 0.0594
R-squared for Edu_x: 0.0854
R-squared for FAS_total_raw_x: 0.1578
R-squared for FAS_total_T_x: 0.2266
R-squared for Animals_raw_x: 0.0258
R-squared for Animals_T_x: 0.1692
R-squared for BNT_totalwstim_raw_x: 0.2587
R-squared for BNT_totalwstim_T_x: 0.1499

Model Summary:
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 128)               3200      
                                                                 
 dropout (Dropout)           (None, 128)               0         
                                                                 
 dense_1 (Dense)             (None, 64)                8256      
                                                                 
 dropout_1 (Dropout)         (None, 64)                0         
                                                                 
 dense_2 (Dense)             (None, 32)                2080      
                                                                 
 dense_3 (Dense)             (None, 8)                 264       
                                                                 
=================================================================
Total params: 13,800
Trainable params: 13,800
Non-trainable params: 0
_________________________________________________________________


---------------------------  12-17-24 Tue  ----------------------------------
1. Dataset Augmentation에 대해서 공부
내가 가진 tabula data (means data organized in rows and columns, like what you'd see in a spreadsheet or database table):
- SMOTE (synthetic Minority Over-sampling technique) 
   Create synthetic samples by interpolating between existing data points
   Great for balanced dataset creation
   Works ell with numerical values
- ADASYN (Adaptive Synthetic)
- Random over-sampling
  Randomly duplicate existing records
  Easiest methods but can lead to overfitting
- Adding Gaussian noise to numerical features
  Add slight Gaussian noise to your exsiting values
  Keeps the overall distribution while creating new samples
  Make sure the noise level makes sense for your data range
- Linear interpolation
  Create new samples by interpolating between pairs of exsiting samples
  Helps maintain relationships between features

2. 내가 가진 dataset Summary:
My output variables (NPTs) include:
  Demographic data: Age and Education
  Language tests: FAS (verbal fluency), Animals (semantic fluency), and BNT (Boston Naming Test), both raw and T-scores

My input variables are brain region volumes, focusing on areas involved in language control:
  DLPFC (executive function and working memory)
  Inferior frontal regions (including Broca's area)
  Temporal regions (language comprehension)
  Motor and pre-motor areas
  Subcortical structures (Caudate)
  ACC (cognitive control)

3. 1번중에 어떤 방식으로 augmenting할 지 결정하기에 앞서 data distribution을 보기로 함
0-basic-statistics-distribution_12_17_24.py
distribution, skeness 확인후 두가지 방법으로 augmenting하기로 함


4. 이렇게 visualize해보니 group별로 matrix를 그려보고 싶었음
0-1-basic-statistics-distribution-by-group_12_17_24.py
['syndrome_v2_v2_x'] variables있음 거기서
group_names = {1: 'MCI', 2: 'Dementia', 3: 'NC'}

5. 3, 4 번 결과가 매우 신기함
그래서 print되는 것중 신기한것은 GoogleDrive 2004 folder 에 bgbridge/nn/PreNN-Correlation-Coefficient_Very_Interesting 에 폴더를 만들고 pre analysis라는 ppt로 저장함
가장 신기했던 것 중에 하나는 Caudate은 그룹 별로 봤을때 Dementia그룹에서는 Age와 관련이 없다. 그런데 MCI에서는 상관관계가 있다. 

6. 5-1-MultivariateNormalDataAugmentation_12_17_24.py를 통해
using multivariate normal distribution Data를 augmenting했다.
386 -> 772 개로 augmenting
5_1_augmented_data_multivariate_12_17_24.csv 로 저장

7. 5_1_augmented_data_multivariate_12_17_24.csv 로 neural network 돌려봄
에러남 내일 고쳐야함
6-1-nn_regression_standardscaler_aug_LanControlRegionsOnlymultivariate_12_17_24.py

---------------------------  12-18-24 Wed  ----------------------------------