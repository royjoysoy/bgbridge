# 12-24-24 Korea Roy Seo
# preprocessing:
# 관련 뇌 영역 선택
# 결측치 확인 및 처리
# 데이터 스케일링
# train, evaluation:
# 데이터 분할 (학습/테스트)
# 모델 생성 및 학습
# 학습 과정 시각화
# 테스트 세트에서 성능 평가
# results 
# 학습된 모델 저장
# 학습 과정 그래프 저장

import tensorflow as tf
from tensorflow.keras import layers, models
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# 파일 읽기
file_path = '1__6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv'
df = pd.read_csv(file_path)

def create_cognitive_reserve_network(input_shape):
    # 입력 레이어 - 동적 shape 사용
    brain_input = layers.Input(shape=(input_shape,), name='brain_regions')
    edu_input = layers.Input(shape=(1,), name='education')
    age_input = layers.Input(shape=(1,), name='age')

    # 뇌 영역 처리 서브네트워크
    brain_x = layers.Dense(32, activation='relu')(brain_input)
    brain_x = layers.Dropout(0.3)(brain_x)
    brain_x = layers.Dense(16, activation='relu')(brain_x)

    # 인구통계학적 데이터 처리
    demo_concat = layers.Concatenate()([edu_input, age_input])
    demo_x = layers.Dense(4, activation='relu')(demo_concat)

    # 특징 통합
    combined = layers.Concatenate()([brain_x, demo_x])
    x = layers.Dense(16, activation='relu')(combined)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(8, activation='relu')(x)

    # 출력 레이어
    outputs = [
        layers.Dense(1, name='fas_output')(x),
        layers.Dense(1, name='animals_output')(x),
        layers.Dense(1, name='bnt_output')(x)
    ]

    model = models.Model(
        inputs=[brain_input, edu_input, age_input],
        outputs=outputs
    )
    
    return model

def prepare_data(df):
    # 입력 특징 준비
    brain_cols = [col for col in df.columns if any(region in col.lower() for region in 
        ['frontal', 'temporal', 'parietal', 'caudate', 'cingulate'])]
    
    print("Selected brain regions:", brain_cols)
    print(f"Number of brain regions: {len(brain_cols)}")
    
    X_brain = df[brain_cols].values
    X_edu = df['Edu_x'].values.reshape(-1, 1)
    X_age = df['Age_x'].values.reshape(-1, 1)
    
    # 출력 변수 준비
    y_fas = df['FAS_total_T_x'].values
    y_animals = df['Animals_T_x'].values
    y_bnt = df['BNT_totalwstim_T_x'].values
    
    # 결측치 확인
    print("\nMissing values check:")
    print("Brain features:", np.isnan(X_brain).sum())
    print("Education:", np.isnan(X_edu).sum())
    print("Age:", np.isnan(X_age).sum())
    print("FAS:", np.isnan(y_fas).sum())
    print("Animals:", np.isnan(y_animals).sum())
    print("BNT:", np.isnan(y_bnt).sum())
    
    # 결측치 처리
    X_brain = np.nan_to_num(X_brain, nan=0)
    X_edu = np.nan_to_num(X_edu, nan=np.nanmean(X_edu))
    X_age = np.nan_to_num(X_age, nan=np.nanmean(X_age))
    y_fas = np.nan_to_num(y_fas, nan=np.nanmean(y_fas))
    y_animals = np.nan_to_num(y_animals, nan=np.nanmean(y_animals))
    y_bnt = np.nan_to_num(y_bnt, nan=np.nanmean(y_bnt))
    
    # 스케일링
    scaler = StandardScaler()
    X_brain_scaled = scaler.fit_transform(X_brain)
    X_edu_scaled = scaler.fit_transform(X_edu)
    X_age_scaled = scaler.fit_transform(X_age)
    
    return X_brain_scaled, X_edu_scaled, X_age_scaled, y_fas, y_animals, y_bnt

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('9-training_history_12_24_24.png')
    plt.close()

def main():
    print("데이터 로딩 중...")
    X_brain, X_edu, X_age, y_fas, y_animals, y_bnt = prepare_data(df)
    
    print("\n데이터 분할 중...")
    indices = np.arange(len(X_brain))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    print("\n모델 생성 중...")
    # 실제 brain features 수에 맞춰 모델 생성
    input_shape = X_brain.shape[1]
    model = create_cognitive_reserve_network(input_shape)
    model.compile(
        optimizer='adam',
        loss={
            'fas_output': 'mse',
            'animals_output': 'mse',
            'bnt_output': 'mse'
        }
    )
    
    print("\n모델 구조:")
    model.summary()
    
    print("\n모델 학습 시작...")
    history = model.fit(
        [X_brain[train_idx], X_edu[train_idx], X_age[train_idx]],
        [y_fas[train_idx], y_animals[train_idx], y_bnt[train_idx]],
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    plot_training_history(history)
    
    print("\n테스트 세트에서 성능 평가 중...")
    test_results = model.evaluate(
        [X_brain[test_idx], X_edu[test_idx], X_age[test_idx]],
        [y_fas[test_idx], y_animals[test_idx], y_bnt[test_idx]],
        verbose=0
    )
    
    print("\n테스트 결과:")
    print(f"Total loss: {test_results[0]:.4f}")
    print(f"FAS loss: {test_results[1]:.4f}")
    print(f"Animals loss: {test_results[2]:.4f}")
    print(f"BNT loss: {test_results[3]:.4f}")
    
    # 모델 저장
    model.save('9-cognitive_reserve_model_12_24_24.h5')
    print("\n모델이 '9-cognitive_reserve_model_12_24_24.h5'로 저장되었습니다.")


def evaluate_performance(y_true, y_pred, test_name):
    """
    회귀 모델의 성능을 다양한 지표로 평가
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{test_name} 성능 평가:")
    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def plot_prediction_vs_actual(y_true, y_pred, test_name):
    """
    실제값과 예측값의 산점도 생성
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{test_name}: Prediction vs Actual')
    plt.tight_layout()
    plt.savefig(f'9-prediction_vs_actual_{test_name}_12_24_24.png')
    plt.close()

def main():
    print("데이터 로딩 중...")
    X_brain, X_edu, X_age, y_fas, y_animals, y_bnt = prepare_data(df)
    
    print("\n데이터 분할 중...")
    indices = np.arange(len(X_brain))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    print("\n모델 생성 중...")
    input_shape = X_brain.shape[1]
    model = create_cognitive_reserve_network(input_shape)
    model.compile(
        optimizer='adam',
        loss={
            'fas_output': 'mse',
            'animals_output': 'mse',
            'bnt_output': 'mse'
        },
        metrics=['mae', 'mse']  # 추가 메트릭
    )
    
    print("\n모델 구조:")
    model.summary()
    
    print("\n모델 학습 시작...")
    history = model.fit(
        [X_brain[train_idx], X_edu[train_idx], X_age[train_idx]],
        [y_fas[train_idx], y_animals[train_idx], y_bnt[train_idx]],
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # 예측 수행
    y_pred = model.predict([X_brain[test_idx], X_edu[test_idx], X_age[test_idx]])
    
    # 각 출력에 대한 성능 평가
    test_names = ['FAS', 'Animals', 'BNT']
    y_true_list = [y_fas[test_idx], y_animals[test_idx], y_bnt[test_idx]]
    
    performance_results = {}
    for i, (test_name, y_true, pred) in enumerate(zip(test_names, y_true_list, y_pred)):
        # 성능 평가
        metrics = evaluate_performance(y_true, pred, test_name)
        performance_results[test_name] = metrics
        
        # 시각화
        plot_prediction_vs_actual(y_true, pred, test_name)
    
    # 결과를 DataFrame으로 정리
    metrics_df = pd.DataFrame({
        'Test': test_names,
        'R²': [performance_results[test]['r2'] for test in test_names],
        'RMSE': [performance_results[test]['rmse'] for test in test_names],
        'MAE': [performance_results[test]['mae'] for test in test_names]
    })
    
    print("\n전체 성능 요약:")
    print(metrics_df.to_string(index=False))
    
    # 결과 저장
    metrics_df.to_csv('9-model_performance_metrics_12_24_24.csv', index=False)
    print("\n성능 지표가 '9-model_performance_metrics_12_24_24.csv'로 저장되었습니다.")
    
    model.save('9-cognitive_reserve_model_12_24_24.h5')
    print("\n모델이 '9-cognitive_reserve_model_12_24_24.h5'로 저장되었습니다.")

if __name__ == "__main__":
    main()