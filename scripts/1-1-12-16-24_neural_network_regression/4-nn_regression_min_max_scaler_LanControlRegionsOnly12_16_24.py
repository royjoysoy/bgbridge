# 12-16-24 Mon In Korea Time Roy Seo
# The 2nd attempt to run Neural Network Regression
# Input: lh.aparc.stats, rh.aparc.stats Output: NPTs
# basic script 2 (w/ MinMaxScaler)


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler  # StandardScaler 대신 MinMaxScaler 사용
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Read the data
file_path = '/Users/test_terminal/Desktop/bgbridge/scripts/1-1-12-16-24_neural_network_regression/1__6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv'
df = pd.read_csv(file_path)

# Define input and output variables (기존 변수 정의 유지)
output_NPT = [
    'Age_x', 'Edu_x',
    'FAS_total_raw_x', 'FAS_total_T_x',
    'Animals_raw_x', 'Animals_T_x',
    'BNT_totalwstim_raw_x', 'BNT_totalwstim_T_x'
]

input_region = [ 
# Left and Right DLPFC (Dorsolateral Prefrontal Cortex):
'rostralmiddlefrontal', 'rostralmiddlefrontal_rh',
'caudalmiddlefrontal', 'caudalmiddlefrontal_rh',
# LH, RH inferior frontal:
'parsopercularis', 'parsopercularis_rh',
'parsorbitalis', 'parsorbitalis_rh',
'parstriangularis', 'parstriangularis_rh',
# LH, RH Lateral orbitofrontal:
'lateralorbitofrontal', 'lateralorbitofrontal_rh',
# LH, RH Middle Temporal:
'middletemporal', 'middletemporal_rh',
# Pre-SMA (Pre-Supplementary Motor Area):
'superiorfrontal', 'superiorfrontal_rh', # (부분적으로 포함)
# LH, RH Precentral:
'precentral', 'precentral_rh',
# LH, RH Caudate:
'Left-Caudate', 'Right-Caudate',
# ACC (Anterior Cingulate Cortex):
'rostralanteriorcingulate', 'rostralanteriorcingulate_rh',
'caudalanteriorcingulate', 'caudalanteriorcingulate_rh'
]
# Prepare input (X) and output (y) data
X = df[input_region].values
y = df[output_NPT].values

# Handle missing values
X = np.nan_to_num(X, nan=np.nanmean(X))
y = np.nan_to_num(y, nan=np.nanmean(y))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Min-Max scaling (0-1 정규화)
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# 스케일러 저장 (나중에 역변환을 위해)
import joblib
joblib.dump(scaler_X, 'scaler_X.save')
joblib.dump(scaler_y, 'scaler_y.save')

# Create the model (기존 모델 구조 유지)
model = Sequential([
    Dense(128, activation='relu', input_shape=(len(input_region),)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(output_NPT))
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train_scaled, y_train_scaled,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate the model
train_loss = model.evaluate(X_train_scaled, y_train_scaled, verbose=0)
test_loss = model.evaluate(X_test_scaled, y_test_scaled, verbose=0)

print("\nTraining Loss:", train_loss)
print("Test Loss:", test_loss)

# Make predictions and inverse transform to original scale
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_original = scaler_y.inverse_transform(y_test_scaled)

# Calculate R-squared for each output variable
r2_scores = []
for i in range(len(output_NPT)):
    correlation_matrix = np.corrcoef(y_test_original[:, i], y_pred[:, i])
    r2 = correlation_matrix[0, 1] ** 2
    r2_scores.append(r2)
    print(f"R-squared for {output_NPT[i]}: {r2:.4f}")

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
plt.savefig('training_history.png')
plt.close()

# Save results
results = pd.DataFrame({
    'Variable': output_NPT,
    'R_squared': r2_scores
})
results.to_csv('model_performance.csv', index=False)

# Print model summary
print("\nModel Summary:")
model.summary()

# Save the model
model.save('brain_cognitive_model.h5')


'''
  -- Results --
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