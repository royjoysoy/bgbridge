# 12-16-24 Mon In Korea Time Roy Seo
# The first attempt to run Neural Network Regression
# Input: lh.aparc.stats, rh.aparc.stats Output: NPTs
# basic script (w/ standard scalers)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Read the data
file_path = '/Users/test_terminal/Desktop/bgbridge/scripts/1-1-12-16-24_neural_network_regression/1__6-merged_NPT_w_o_outliers__voxel_w_outliers_12_12_24_manually_cleaned_the_duplicated_rows_12_13_24.csv'
df = pd.read_csv(file_path)

# Define input (y_vars) and output (output_NPT) variables
output_NPT = [
    'Age_x', 'Edu_x',
    'FAS_total_raw_x', 'FAS_total_T_x',
    'Animals_raw_x', 'Animals_T_x',
    'BNT_totalwstim_raw_x', 'BNT_totalwstim_T_x'
]

input_region = [ 
    # Original cortical regions - Left hemisphere # lh.aparc.stats & rh.aparc.stats
    'bankssts', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus',
    'entorhinal', 'fusiform', 'inferiorparietal', 'inferiortemporal',
    'isthmuscingulate', 'lateraloccipital', 'lateralorbitofrontal', 'lingual',
    'medialorbitofrontal', 'middletemporal', 'parahippocampal', 'paracentral',
    'parsopercularis', 'parsorbitalis', 'parstriangularis', 'pericalcarine',
    'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
    'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal',
    'superiorparietal', 'superiortemporal', 'supramarginal', 'frontalpole',
    'temporalpole', 'transversetemporal', 'insula',
    
    # Original cortical regions - Right hemisphere
    'bankssts_rh', 'caudalanteriorcingulate_rh', 'caudalmiddlefrontal_rh', 
    'cuneus_rh', 'entorhinal_rh', 'fusiform_rh', 'inferiorparietal_rh',
    'inferiortemporal_rh', 'isthmuscingulate_rh', 'lateraloccipital_rh',
    'lateralorbitofrontal_rh', 'lingual_rh', 'medialorbitofrontal_rh',
    'middletemporal_rh', 'parahippocampal_rh', 'paracentral_rh',
    'parsopercularis_rh', 'parsorbitalis_rh', 'parstriangularis_rh',
    'pericalcarine_rh', 'postcentral_rh', 'posteriorcingulate_rh',
    'precentral_rh', 'precuneus_rh', 'rostralanteriorcingulate_rh',
    'rostralmiddlefrontal_rh', 'superiorfrontal_rh', 'superiorparietal_rh',
    'superiortemporal_rh', 'supramarginal_rh', 'frontalpole_rh',
    'temporalpole_rh', 'transversetemporal_rh', 'insula_rh',
    
    # Additional subcortical and other regions # aseg.stats
    'Left-Lateral-Ventricle', 'Left-Inf-Lat-Vent', 'Left-Cerebellum-White-Matter',
    'Left-Cerebellum-Cortex', 'Left-Thalamus-Proper', 'Left-Caudate',
    'Left-Putamen', 'Left-Pallidum', '3rd-Ventricle', '4th-Ventricle',
    'Brain-Stem', 'Left-Hippocampus', 'Left-Amygdala', 'CSF',
    'Left-Accumbens-area', 'Left-VentralDC', 'Left-vessel',
    'Left-choroid-plexus', 'Right-Lateral-Ventricle', 'Right-Inf-Lat-Vent',
    'Right-Cerebellum-White-Matter', 'Right-Cerebellum-Cortex',
    'Right-Thalamus-Proper', 'Right-Caudate', 'Right-Putamen', 'Right-Pallidum',
    'Right-Hippocampus', 'Right-Amygdala', 'Right-Accumbens-area',
    'Right-VentralDC', 'Right-vessel', 'Right-choroid-plexus', '5th-Ventricle',
    'WM-hypointensities', 'Left-WM-hypointensities', 'Right-WM-hypointensities',
    'non-WM-hypointensities', 'Left-non-WM-hypointensities',
    'Right-non-WM-hypointensities', 'Optic-Chiasm', 'CC_Posterior',
    'CC_Mid_Posterior', 'CC_Central', 'CC_Mid_Anterior', 'CC_Anterior'
]

# Prepare input (X) and output (y) data
X = df[input_region].values  # Brain regions as input
y = df[output_NPT].values  # Cognitive measures as output

# Handle missing values
# Replace NaN with mean for both input and output
X = np.nan_to_num(X, nan=np.nanmean(X))
y = np.nan_to_num(y, nan=np.nanmean(y))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# Create the model
model = Sequential([
    Dense(128, activation='relu', input_shape=(len(input_region),)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(len(output_NPT))  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
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

# Make predictions
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# Calculate R-squared for each output variable
r2_scores = []
for i in range(len(output_NPT)):
    correlation_matrix = np.corrcoef(y_test[:, i], y_pred[:, i])
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

# Save model performance metrics
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