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


---------------------------  12-17-24 Tues  ----------------------------------