import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# Read the augmented data
file_path = '5_1_augmented_data_multivariate_12_17_24.csv'
df = pd.read_csv(file_path)



# Define input and output variables
input_region = [
    'rostralmiddlefrontal', 'rostralmiddlefrontal_rh',
    'caudalmiddlefrontal', 'caudalmiddlefrontal_rh',
    'parsopercularis', 'parsopercularis_rh',
    'parsorbitalis', 'parsorbitalis_rh',
    'parstriangularis', 'parstriangularis_rh',
    'lateralorbitofrontal', 'lateralorbitofrontal_rh',
    'middletemporal', 'middletemporal_rh',
    'superiorfrontal', 'superiorfrontal_rh',
    'precentral', 'precentral_rh',
    'Left-Caudate', 'Right-Caudate',
    'rostralanteriorcingulate', 'rostralanteriorcingulate_rh',
    'caudalanteriorcingulate', 'caudalanteriorcingulate_rh'
]

output_NPT = [
    'FAS_total_raw_x', 'FAS_total_T_x',
    'Animals_raw_x', 'Animals_T_x',
    'BNT_totalwstim_raw_x', 'BNT_totalwstim_T_x'
]



# missing value check
print("Missing values before preprocessing:")
print(df[input_region + output_NPT].isnull().sum())


# 결측치 처리 방법 1: 평균값으로 대체
# df[input_region] = df[input_region].fillna(df[input_region].mean())
# df[output_NPT] = df[output_NPT].fillna(df[output_NPT].mean())

# 또는 결측치 처리 방법 2: 중앙값으로 대체 (이상치의 영향을 덜 받음)
df[input_region] = df[input_region].fillna(df[input_region].median())
df[output_NPT] = df[output_NPT].fillna(df[output_NPT].median())

# 결측치 처리 후 확인
print("\nMissing values after preprocessing:")
print(df[input_region + output_NPT].isnull().sum())

# 결측치 처리 후 데이터 통계 확인
print("\nData statistics after proprocessing:")
print(df[input_region + output_NPT].describe())

# Prepare the data
X = df[input_region]
y = df[output_NPT]


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Build the model
def build_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(len(output_NPT), activation='linear')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
                 loss='mse',
                 metrics=['mae'])
    return model

# Train the model
model = build_model(len(input_region))

history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Evaluate the model
train_predictions_scaled = model.predict(X_train_scaled)
test_predictions_scaled = model.predict(X_test_scaled)

# Inverse transform predictions
train_predictions = scaler_y.inverse_transform(train_predictions_scaled)
test_predictions = scaler_y.inverse_transform(test_predictions_scaled)

# Calculate R-squared for each output variable
def calculate_r2(y_true, y_pred):
    r2_scores = {}
    for i, col in enumerate(output_NPT):
        ss_res = np.sum((y_true[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_true[:, i] - np.mean(y_true[:, i])) ** 2)
        r2_scores[col] = 1 - (ss_res / ss_tot)
    return r2_scores

# Calculate R-squared scores
train_r2 = calculate_r2(y_train.values, train_predictions)
test_r2 = calculate_r2(y_test.values, test_predictions)

# Print results
print("\nTraining R-squared scores:")
for var, score in train_r2.items():
    print(f"{var}: {score:.4f}")

print("\nTest R-squared scores:")
for var, score in test_r2.items():
    print(f"{var}: {score:.4f}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig('6-1_training_history_12_17_24.png')
plt.close()

# Save the model
model.save('6-1_neural_network_model_12_17_24.h5')

# Plot actual vs predicted for each output variable
plt.figure(figsize=(20, 4))
for i, col in enumerate(output_NPT):
    plt.subplot(1, len(output_NPT), i+1)
    plt.scatter(y_test[col], test_predictions[:, i], alpha=0.5)
    plt.plot([y_test[col].min(), y_test[col].max()], 
             [y_test[col].min(), y_test[col].max()], 
             'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{col}\nR² = {test_r2[col]:.4f}')

plt.tight_layout()
plt.savefig('6-1_prediction_scatter_plots_12_17_24.png')
plt.close()