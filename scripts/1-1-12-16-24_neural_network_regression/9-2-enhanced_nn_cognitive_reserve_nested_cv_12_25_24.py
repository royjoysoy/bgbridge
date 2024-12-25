# 12-25-24 Roy Seo
# 전에 9-1-enhanced_nn_cognitive_reserve_ensemble_12_25_24.py script 앙상블 예측결과, 콘솔에 찍히는 결과의 불일치를 줄이기 위해 코드를 개선)

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
    신경망 모델 생성
    """
    inputs = layers.Input(shape=(input_shape,), name=f'{name}_input')
    
    x = layers.Dense(32, activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(16, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    
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
    df_clean = df.copy()
    df_clean[target_col] = df_clean[target_col].fillna(df_clean[target_col].median())
    
    # 이상치 처리
    z_threshold = 3
    for col in brain_cols:
        z_scores = np.abs(stats.zscore(df_clean[col], nan_policy='omit'))
        df_clean[col] = df_clean[col].mask(z_scores > z_threshold, df_clean[col].median())
    
    # 특징 선택
    correlations = df_clean[brain_cols].corrwith(df_clean[target_col]).abs()
    selected_features = correlations[correlations > 0.1].index.tolist()
    
    if not selected_features:
        selected_features = brain_cols
    
    print(f"\n{target_col}에 대한 선택된 특징 수: {len(selected_features)}")
    
    X = df_clean[selected_features].values
    y = df_clean[target_col].values
    
    return X, y, selected_features

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
                patience=20,
                restore_best_weights=True
            )
            
            model.fit(
                X_train_scaled, y_train,
                epochs=200,
                batch_size=32,
                validation_data=(X_val_scaled, y_val),
                callbacks=[early_stopping],
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
    결과 시각화
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    plt.plot(y_true, p(y_true), "r--", alpha=0.8)
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('실제 값')
    plt.ylabel('예측 값')
    plt.title(f'{title}: 예측 vs 실제')
    
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}',
             transform=plt.gca().transAxes, 
             verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    print("데이터 로딩 중...")
    df = pd.read_csv('1__7-merged_dataset_12_25_24.csv')
    
    # 뇌 영역 컬럼 선택
    brain_cols = [col for col in df.columns if any(region in col.lower() for region in 
        ['frontal', 'temporal', 'parietal', 'caudate', 'cingulate'])]
    
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
        plot_results(y, predictions, test_name, f'9-2-nested_cv_{test_name.lower()}_results_12_25_24.png')
    
    # 결과를 DataFrame으로 저장
    results_df = pd.DataFrame(results)
    results_df.to_csv('9-2-nested_cv_results_12_25_24.csv', index=False)
    print("\n최종 결과가 '9-2-nested_cv_results_12_25_24.csv'에 저장되었습니다.")

if __name__ == "__main__":
    main()