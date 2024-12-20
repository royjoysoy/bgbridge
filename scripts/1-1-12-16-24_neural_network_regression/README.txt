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

---------------------------  12-20-24 Fri  ----------------------------------
1. 학습이 되지 않고  다 NaN으로 찍히는 에러
- 내가 생각한 에러의 첫번째 이유는 현재 script에서는 missing data points에 대한 명시적인 처리가 없다. 
- missing data 를 median (chosen for robustness to outliers instead of mean) 으로 대체함
NaN error가 해결됨

2. NaN error를 고치고나서 overfitting으로 보이는 현상이 나타남 (train loss 는 줄지만, validation loss는 줄지 않음)
- Loss의 초기값:
첫 에폭에서 training loss가 1.0603에서 시작, validation loss는 0.9749에서 시작
- Loss의 변화 패턴:
Training loss: 1.0603 → 0.4456 (약 58% 감소), Validation loss: 0.9749 → 0.8915 (약 9% 감소)
- Overfitting 징후:
Training loss는 계속 감소하는데 validation loss는 거의 감소하지 않고 오히려 약간 증가하는 경향을 보임, 이는 전형적인 overfitting의 징후
약 16-20 에폭 이후부터 validation loss가 no longer improved
- 에폭 수 증가에 대해:
overfitting이 이미 발생하고 있기 때문에, 에폭을 늘리면 overfitting만 더 심해질 것으로 보임
- Validation loss가 감소하지 않는 이유:
  1: 모델이 너무 복잡할 수 있다. (현재 128-64-32 뉴런 구조)
  2: Dropout (0.3, 0.2)이 있지만 충분하지 않을 수 있습니다 (dropout:  학습 과정에서 무작위로 일정 비율의 뉴런을 끄는것 drop out 0.3 30% 뉴런 끄는 것)
  3: 데이터셋이 작을 수도 있다.

3. 7-1-model-simplication-12-20-24.py
모델 단순화를 한 후의 결과
- Loss 패턴 변화:
  Training loss: 1.6702 → 0.7239 (약 57% 감소)
  Validation loss: 1.0583 → 0.8621 (약 19% 감소)
  이전 버전보다 validation loss의 감소폭이 좀 더 커짐
- R-squared 점수:
  Training R-squared: 0.25~0.39 범위
  Test R-squared: -0.02~0.14 범위
  Animals 관련 변수들의 예측이 상대적으로 더 잘되고 있음
  하지만 여전히 test set에서의 성능이 좋지 않다.
- Overfitting 상태:
  여전히 overfitting이 있지만, 이전 모델보다는 조금 덜함
  Training과 validation loss의 차이가 이전보다 줄었다

- 이 전 모델에서 R²가 음수가 나온 이유:
R² = 1 - (SS_res / SS_tot)
여기서:
SS_res = Σ(y_true - y_pred)²         # 예측값과 실제값의 차이의 제곱합
SS_tot = Σ(y_true - y_true_mean)²    # 실제값과 평균의 차이의 제곱합

    R²가 음수가 되는 경우:
    SS_res > SS_tot 일 때
    즉, 모델의 예측 오차(SS_res)가 단순히 평균값을 사용했을 때의 오차(SS_tot)보다 더 클 때
    예를 들어: 실제 값들: [1, 2, 3, 4, 5] (평균 = 3)
           모델 예측: [0, 0, 6, 6, 6]
           이 경우 평균(3)을 사용하는 것이 더 나은 예측이 됨
이 모델에서 R²가 음수가 나온 이유:

모델이 과적합(overfitting)
복잡한 모델 구조
높은 학습률

4. 7-2-model-simplification-early-stopping-12-20-24.py
model simplication + early stopping 추가
early stopping : training set에 대한 손실 (loss)은 지속적으로 감소하지만, 검증 데이터 (validation set)에 대한 손실은 특정 시점 이후 증가하기 시작할 수 있다. 
                 validation set에 대한 손실은 특정 시점 이후 증가하기 시작할 수 있다. 

                 검증 손실이 더 이상 감소하지 않거나 악화되기 시작하면, 모델이 훈련 데이터에 과적합되고 있음을 나타냄

                 early stopping은 검증 손실이 일정 횟수 (e.g., 5 epochs) 동안 개선되지 않으면 학습을 멈추고 가장 좋은 검증 성늠을 보였던 지점의 모델을 저장한다. 

- 조기 중단 효과:
  학습이 57 에폭에서 자동으로 중단됨 (원래 100 에폭 설정)
  이는 validation loss가 10 에폭 동안 개선되지 않았음을 의미합니다
성능 비교:
Before Early Stopping:
Train R² : 0.25~0.39
Test R² : -0.02~0.14

After Early Stopping:
Train R² : 0.17~0.29
Test R² : 0.05~0.15
주요 관찰:
Overfitting이 약간 감소 (train과 test의 R² 차이가 줄어듦)
Animals 관련 변수들이 여전히 가장 잘 예측됨
Test set의 음수 R² 값들이 없어졌습니다 (모든 값이 양수)

5. 7-3-model-simplification-early-stopping-L2regularization-12-20-24.py
- L2 Regularization 
  기본 개념:
  모델의 가중치(weights)가 너무 큰 값을 가지지 않도록 제한하는 방법
  비용 함수(loss function)에 가중치의 제곱항을 추가
  
  수학적 표현:
  새로운 비용 함수 = 원래 비용 함수 + λ * Σ(가중치²)
  여기서:
  - λ (lambda): regularization 강도를 조절하는 하이퍼파라미터
  - Σ(가중치²): 모든 가중치의 제곱의 합

  작동 방식: 
  # 예시
  Dense(64, kernel_regularizer=l2(0.01))  # 0.01은 λ 값

  큰 가중치는 더 큰 페널티를 받음 (제곱 때문에)
모델이 특정 특성에 과도하게 의존하는 것을 방지
가중치들이 더 균일하게 분포하도록 유도

  효과:
  모델이 더 단순해짐
  특정 훈련 데이터에 과도하게 맞추는 것을 방지
  일반화 성능 향상
  비유:
  지도 학습에서 "너무 열심히 외우지 말고, 적당히 이해하면서 공부하라"고 하는 것과 비슷
  시험 문제와 똑같은 문제만 풀 수 있는 것이 아니라, 비슷한 유형의 다른 문제도 풀 수 있게 됨

L2 Regularization의 λ 값(예제에서는 0.01)은 하이퍼파라미터로, 이 값이:
크면: 더 강한 규제 → 더 단순한 모델
작으면: 더 약한 규제 → 더 복잡한 모델이 가능

6. 7-3-model-simplification-early-stopping-L2regularization-12-20-24.py의 결과
- 모델 성능 변화:
이전 모델 (Early Stopping만)
  Train R² : 0.17~0.29
  Test R² : 0.05~0.15
L2 추가 후
  Train R² : 0.19~0.34
  Test R² : 0.01~0.13
- 관찰된 변화:
  Training R²와 Test R² 간의 차이가 여전히 크다 (overfitting)
  L2 regularization이 overfitting을 완전히 해결하지는 못했다
  Test R²가 오히려 약간 떨어진 변수들이 있다.
- Loss 패턴: 
  초기 loss가 더 높게 시작함 (2.1221)
  이는 L2 regularization이 가중치에 패널티를 부과하기 때문
  validation loss가 더 안정적으로 감소하는 패턴을 보임

  7. 7-4-some-tunning-12-20-24.py
  - L2 regularization 강도조정 : 더 약한 규제 -kernel_regularizer=l2(0.001)  더 강한 주제 kernel_regularizer=l2(0.05) : 둘 다 더 악화
  - Learning Rate 조정 optimizer=Adam(learning_rate=0.001에서 0.0001로 조정) : 악화
  - 추가적인 모델 단순화: 성능 차이 없어보임

  8. 8-neural-plasticity_12_20_24.py
  - 갑자기 neural-plasticity가 궁금해져서 논문의 NPT 와 brain regions 기준이 있는지 찾아보았는데 찾지는 못했다. 임의로 다음과 같이 해봄 
  - MCI, Dementia, NC에서 NPTs 와 Regions of Interest volumen 평균과 std 구한 뒤, 
  - 1.뇌 볼륨: 환자의 진단 그룹(MCI 또는 Dementia) 평균보다 작은 영역이 하나 이상 있음
  - 2. NPT 성적: 정상군(NC) 평균에서 1 표준편차를 뺀 값보다 높은 점수를 2개 이상의 테스트에서 획득
  이 두가지 찾아보는 scripts
  
  outputs:
  
  8-brain_regions_analysis_12_20_24.png
  8-left_hemisphere_12_20_24.png
  8-right_hemisphere_12_20_24.png	
  8-decreased_regions_12_20_24.png				
  8-region_npt_pairs_12_20_24.png
  8-region_npt_pair_statistics_12_20_24.csv
  8-group_statistics_12_20_24.csv			
  8-plasticity_results_detailed_12_20_24.csv	
  