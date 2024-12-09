import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# If I want to adopt a GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)

def prepare_data(df):
    # Drop specified columns
    columns_to_drop = ['V1_x', 'Study_ID', 'patientlist_ethnicbackground_v2_v2_x', 
                      'patientlist_ethnicgroup_v2_v2_x']
    X = df.drop(columns_to_drop + ['syndrome_v2_v2_x'], axis=1)
    y = df['syndrome_v2_v2_x'].values - 1  # Subtract 1 to convert to 0-based indexing
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

class DiagnosisPredictor(nn.Module):
    def __init__(self, input_size):
        super(DiagnosisPredictor, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 output classes
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.layers(x)

def train_model(model, X_train, y_train, epochs=100):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_train_tensor).sum().item() / len(y_train_tensor)
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

def main():
    # Load data
    df = pd.read_csv('/Users/test_terminal/Desktop/bgbridge/scripts/0-2-120924_dataset_merge/2-9-1-aseg-lh-aparc-merged_cleaned.csv')  # Replace with your data loading method
    
    X, y = prepare_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    input_size = X_train.shape[1]
    model = DiagnosisPredictor(input_size)
    
    train_model(model, X_train, y_train)
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test)
        predictions = model(X_test_tensor)
        _, predicted = torch.max(predictions.data, 1)
        accuracy = (predicted == torch.LongTensor(y_test)).sum().item() / len(y_test)
        print(f'Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()