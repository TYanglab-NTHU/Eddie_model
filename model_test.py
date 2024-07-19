import torch
from torch import nn
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import matplotlib.pyplot as plt
from model_train import MyModel

def load_data():
    try:
        with open('data_prepared.pkl', 'rb') as f:
            X_train, X_test, y_train, y_test, value_type_encoder, equilibrium_encoder, scalers = pickle.load(f)
        print("Data loaded successfully")
        return X_train, X_test, y_train, y_test, value_type_encoder, equilibrium_encoder, scalers
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def test_model():
    data = load_data()
    if data is None:
        print("Failed to load data. Please ensure 'data_prepared.pkl' exists and is not corrupted.")
        return

    X_train, X_test, y_train, y_test, value_type_encoder, equilibrium_encoder, scalers = data

    value_type_shape = value_type_encoder.categories_[0].size
    equilibrium_shape = equilibrium_encoder.categories_[0].size
    smiles_shape = 2048

    model = MyModel(value_type_shape, equilibrium_shape, smiles_shape)

    try:
        model.load_state_dict(torch.load('best_model_state_dict.pth'))
        model.eval()
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(X_test_tensor)
        y_pred = y_pred.squeeze().numpy()

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    print(f"Number of training samples: {n_train}")
    print(f"Number of test samples: {n_test}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')

    info_text = (f'Training samples: {n_train}\n'
                 f'Test samples: {n_test}\n'
                 f'MSE: {mse:.4f}\n'
                 f'MAE: {mae:.4f}\n'
                 f'R²: {r2:.4f}')

    plt.text(0.05, 0.95, info_text, 
             transform=plt.gca().transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.savefig('prediction_plot_with_metrics_and_samples.png', dpi=300)
    plt.close()

    print("預測結果圖已保存為 'prediction_plot_with_metrics_and_samples.png'")

if __name__ == '__main__':
    test_model()