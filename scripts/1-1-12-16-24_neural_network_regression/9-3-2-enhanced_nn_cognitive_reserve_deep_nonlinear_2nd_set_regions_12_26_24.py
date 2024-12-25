# 12-26-24 Roy Seo
# 9-2-enhanced_nn_cognitive_reserve_nested_cv_12_25_24.py 에서 비선형성을 강화하고 더 깊은 신경망 구조를 적용하여 개선
# regions of interest를 두 sets을 사용해봄 2번째 set는 9-3-2로 저장

import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 랜덤 시드 설정
np.random.seed(42)
tf.random.set_seed(42)

def create_model(input_shape, name):
    """
    개선된 깊은 신경망 모델 생성
    - 더 깊은 층 구조
    - 다양한 활성화 함수
    - Skip connection 추가
    """
    inputs = layers.Input(shape=(input_shape,), name=f'{name}_input')
    
    # 첫 번째 블록
    x1 = layers.Dense(64, activation='relu')(inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(0.3)(x1)
    
    # 두 번째 블록 (ELU 활성화 함수 사용)
    x2 = layers.Dense(32, activation='elu')(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(0.2)(x2)
    
    # Skip connection
    x2 = layers.Concatenate()([x2, x1])
    
    # 세 번째 블록 (SELU 활성화 함수 사용)
    x3 = layers.Dense(16, activation='selu')(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Dropout(0.1)(x3)
    
    # 출력 레이어
    outputs = layers.Dense(1, name=f'{name}_output',
                         kernel_regularizer=tf.keras.regularizers.l2(0.01))(x3)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f'{name}_model')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='huber',  # Huber loss 사용 (MSE와 MAE의 장점 결합)
        metrics=['mae']
    )
    return model

def prepare_data(df, target_col, brain_cols):
    """
    데이터 전처리 및 특징 엔지니어링
    - 다항식 특징 추가
    - 특징 선택 기준 완화
    """
    df_clean = df.copy()
    df_clean[target_col] = df_clean[target_col].fillna(df_clean[target_col].median())
    
    # 이상치 처리
    z_threshold = 3
    for col in brain_cols:
        z_scores = np.abs(stats.zscore(df_clean[col], nan_policy='omit'))
        df_clean[col] = df_clean[col].mask(z_scores > z_threshold, df_clean[col].median())
    
    # 기본 특징
    correlations = df_clean[brain_cols].corrwith(df_clean[target_col]).abs()
    selected_features = correlations[correlations > 0.05].index.tolist()  # 임계값 완화
    
    if not selected_features:
        selected_features = brain_cols
    
    X = df_clean[selected_features].values
    
    # 다항식 특징 추가 (2차)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    print(f"\n{target_col}에 대한 원본 특징 수: {len(selected_features)}")
    print(f"다항식 특징 추가 후 특징 수: {X_poly.shape[1]}")
    
    y = df_clean[target_col].values
    
    return X_poly, y, selected_features

def nested_cross_validation(X, y, model_name, n_outer=5, n_inner=5):
    """
    Nested Cross Validation 수행
    """
    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=n_inner, shuffle=True, random_state=42)
    
    outer_scores = []
    all_predictions = np.zeros_like(y)
    
    for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        print(f"\n외부 Fold {outer_fold + 1}/{n_outer}")
        
        X_train_outer, X_test = X[train_idx], X[test_idx]
        y_train_outer, y_test = y[train_idx], y[test_idx]
        
        # 내부 CV를 위한 모델들
        inner_models = []
        
        # 내부 Cross Validation
        for inner_fold, (train_idx_inner, val_idx_inner) in enumerate(inner_cv.split(X_train_outer)):
            print(f"내부 Fold {inner_fold + 1}/{n_inner}")
            
            X_train = X_train_outer[train_idx_inner]
            X_val = X_train_outer[val_idx_inner]
            y_train = y_train_outer[train_idx_inner]
            y_val = y_train_outer[val_idx_inner]
            
            # 스케일링
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # 모델 학습
            model = create_model(X_train_scaled.shape[1], f"{model_name}_inner_{inner_fold}")
            
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=30,  # patience 증가
                restore_best_weights=True
            )
            
            # Learning rate 스케줄링 추가
            lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=0.0001
            )
            
            model.fit(
                X_train_scaled, y_train,
                epochs=300,  # epochs 증가
                batch_size=32,
                validation_data=(X_val_scaled, y_val),
                callbacks=[early_stopping, lr_scheduler],
                verbose=0
            )
            
            inner_models.append((model, scaler))
        
        # 외부 테스트 세트에 대한 앙상블 예측
        test_predictions = []
        for model, scaler in inner_models:
            X_test_scaled = scaler.transform(X_test)
            pred = model.predict(X_test_scaled, verbose=0).flatten()
            test_predictions.append(pred)
        
        # 앙상블 예측 평균
        ensemble_predictions = np.mean(test_predictions, axis=0)
        all_predictions[test_idx] = ensemble_predictions
        
        # 외부 폴드 성능 평가
        fold_score = {
            'r2': r2_score(y_test, ensemble_predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_predictions)),
            'mae': mean_absolute_error(y_test, ensemble_predictions)
        }
        outer_scores.append(fold_score)
        
        print(f"외부 Fold {outer_fold + 1} 성능:")
        print(f"R²: {fold_score['r2']:.4f}")
        print(f"RMSE: {fold_score['rmse']:.4f}")
        print(f"MAE: {fold_score['mae']:.4f}")
    
    # 전체 성능 계산
    final_r2 = r2_score(y, all_predictions)
    final_rmse = np.sqrt(mean_squared_error(y, all_predictions))
    final_mae = mean_absolute_error(y, all_predictions)
    
    # 평균 성능 계산
    mean_scores = {
        'r2': np.mean([score['r2'] for score in outer_scores]),
        'rmse': np.mean([score['rmse'] for score in outer_scores]),
        'mae': np.mean([score['mae'] for score in outer_scores])
    }
    
    return mean_scores, (final_r2, final_rmse, final_mae), all_predictions

def plot_results(y_true, y_pred, title, filename):
    """
    결과 시각화 개선
    """
    plt.figure(figsize=(12, 7))
    
    # 산점도
    plt.scatter(y_true, y_pred, alpha=0.5, label='예측값')
    
    # 추세선
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), "r--", alpha=0.8, label='추세선')
    
    # 이상적인 예측선
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='이상적인 예측')
    
    plt.xlabel('실제 값')
    plt.ylabel('예측 값')
    plt.title(f'{title}: 예측 vs 실제')
    plt.legend()
    
    # 성능 지표
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    stats_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
    plt.text(0.05, 0.95, stats_text,
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    print("데이터 로딩 중...")
    df = pd.read_csv('1__7-merged_dataset_12_25_24.csv')
    
    # # 뇌 영역 컬럼 선택
    # brain_cols = [col for col in df.columns if any(region in col.lower() for region in 
    #     ['frontal', 'temporal', 'parietal', 'caudate', 'cingulate'])]

    # 특정 뇌 영역 선택
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
    
    # 컬럼 존재 확인
    available_cols = [col for col in input_region if col in df.columns]
    if len(available_cols) != len(input_region):
        missing_cols = set(input_region) - set(available_cols)
        print(f"경고: 다음 컬럼들이 데이터셋에 없습니다: {missing_cols}")
    
    brain_cols = available_cols
    
    # 인지 검사
    cognitive_tests = {
        'FAS': 'FAS_total_T_x',
        'Animals': 'Animals_T_x',
        'BNT': 'BNT_totalwstim_T_x'
    }
    
    results = []
    for test_name, target_col in cognitive_tests.items():
        print(f"\n{test_name} 모델 학습 시작...")
        
        X, y, selected_features = prepare_data(df, target_col, brain_cols)
        
        # Nested CV 수행
        mean_scores, final_scores, predictions = nested_cross_validation(X, y, test_name)
        
        print(f"\n{test_name} 최종 성능:")
        print(f"평균 Cross-Validation R²: {mean_scores['r2']:.4f}")
        print(f"최종 전체 데이터 R²: {final_scores[0]:.4f}")
        
        # 결과 저장
        results.append({
            'Test': test_name,
            'Mean_CV_R2': mean_scores['r2'],
            'Mean_CV_RMSE': mean_scores['rmse'],
            'Mean_CV_MAE': mean_scores['mae'],
            'Final_R2': final_scores[0],
            'Final_RMSE': final_scores[1],
            'Final_MAE': final_scores[2]
        })
        
        # 결과 시각화
        plot_results(y, predictions, test_name, f'9-3-2-deep_nonlinear_{test_name.lower()}_results_12_26_24.png')
    
    # 결과를 DataFrame으로 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv('9-3-2-deep_nonlinear_results_12_26_24.csv', index=False)
    print("\n최종 결과가 '9-3-2-deep_nonlinear_results_12_26_24.csv'에 저장되었습니다.")

if __name__ == "__main__":
    main()