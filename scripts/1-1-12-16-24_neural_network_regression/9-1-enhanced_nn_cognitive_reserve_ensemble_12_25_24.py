import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# 랜덤 시드 설정
np.random.seed(42)
tf.random.set_seed(42)

def create_model(input_shape, name):
    """
    단순화된 모델 아키텍처
    """
    inputs = layers.Input(shape=(input_shape,), name=f'{name}_input')
    
    # 첫 번째 레이어 블록
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # 두 번째 레이어 블록
    x = layers.Dense(16, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
    # 출력 레이어
    outputs = layers.Dense(1, name=f'{name}_output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name=f'{name}_model')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

def prepare_data(df, target_col, brain_cols):
    """
    데이터 전처리 및 특징 선택
    """
    # 결측치 처리
    df_clean = df.copy()
    df_clean[target_col] = df_clean[target_col].fillna(df_clean[target_col].median())
    
    # 이상치 처리 (Z-score method)
    z_threshold = 3
    for col in brain_cols:
        z_scores = np.abs(stats.zscore(df_clean[col], nan_policy='omit'))
        df_clean[col] = df_clean[col].mask(z_scores > z_threshold, df_clean[col].median())
    
    # 특징 선택 (상관관계 기반)
    correlations = df_clean[brain_cols].corrwith(df_clean[target_col]).abs()
    selected_features = correlations[correlations > 0.1].index.tolist()
    
    if not selected_features:  # 상관관계가 너무 낮은 경우 모든 특징 사용
        selected_features = brain_cols
    
    print(f"\n{target_col}에 대한 선택된 특징 수: {len(selected_features)}")
    
    # 데이터 준비
    X = df_clean[selected_features].values
    y = df_clean[target_col].values
    
    return X, y, selected_features

def train_and_evaluate(X, y, model_name, n_splits=5):
    """
    교차 검증을 통한 모델 학습 및 평가
    """
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    models = []
    
    scaler = StandardScaler()
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n훈련 Fold {fold + 1}/{n_splits}")
        
        # 데이터 분할
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 스케일링
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # 모델 생성 및 학습
        model = create_model(X_train_scaled.shape[1], model_name)
        
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train_scaled, y_train,
            epochs=200,
            batch_size=32,
            validation_data=(X_val_scaled, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # 성능 평가
        y_pred = model.predict(X_val_scaled, verbose=0).flatten()
        fold_score = {
            'r2': r2_score(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
            'mae': mean_absolute_error(y_val, y_pred)
        }
        fold_scores.append(fold_score)
        models.append((model, scaler))
        
        print(f"Fold {fold + 1} - R²: {fold_score['r2']:.4f}, RMSE: {fold_score['rmse']:.4f}")
    
    return models, fold_scores

def plot_results(y_true, y_pred, title, filename):
    """
    결과 시각화
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # 추세선
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), "r--", alpha=0.8)
    
    # 이상적인 예측선
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title}: Prediction vs Actual')
    
    # 통계 정보 추가
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
             transform=plt.gca().transAxes, 
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # 데이터 로드
    print("데이터 로딩 중...")
    file_path = '1__6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv'
    df = pd.read_csv(file_path)
    
    # 뇌 영역 컬럼 선택
    brain_cols = [col for col in df.columns if any(region in col.lower() for region in 
        ['frontal', 'temporal', 'parietal', 'caudate', 'cingulate'])]
    
    # 각 인지 검사별 모델 학습
    cognitive_tests = {
        'FAS': 'FAS_total_T_x',
        'Animals': 'Animals_T_x',
        'BNT': 'BNT_totalwstim_T_x'
    }
    
    results = {}
    for test_name, target_col in cognitive_tests.items():
        print(f"\n{test_name} 모델 학습 시작...")
        
        # 데이터 준비
        X, y, selected_features = prepare_data(df, target_col, brain_cols)
        
        # 모델 학습 및 평가
        models, fold_scores = train_and_evaluate(X, y, test_name)
        
        # 최종 성능 계산
        mean_scores = {
            metric: np.mean([score[metric] for score in fold_scores])
            for metric in ['r2', 'rmse', 'mae']
        }
        
        results[test_name] = {
            'models': models,
            'scores': mean_scores,
            'features': selected_features
        }
        
        print(f"\n{test_name} 평균 성능:")
        print(f"R²: {mean_scores['r2']:.4f}")
        print(f"RMSE: {mean_scores['rmse']:.4f}")
        print(f"MAE: {mean_scores['mae']:.4f}")
        
        # 전체 데이터에 대한 예측 및 시각화
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_pred = np.mean([model.predict(X_scaled, verbose=0).flatten() 
                         for model, _ in models], axis=0)
        
        plot_results(y, y_pred, test_name, f'9-1-{test_name.lower()}_results_12_25_24.png')
    
    # 결과를 DataFrame으로 저장
    results_df = pd.DataFrame({
        'Test': list(results.keys()),
        'R²': [results[test]['scores']['r2'] for test in results],
        'RMSE': [results[test]['scores']['rmse'] for test in results],
        'MAE': [results[test]['scores']['mae'] for test in results]
    })
    
    results_df.to_csv('9-1-cognitive_model_results_12_25_24.csv', index=False)
    print("\n최종 결과가 '9-1-cognitive_model_results_12_25_24.csv'에 저장되었습니다.")

if __name__ == "__main__":
    main()