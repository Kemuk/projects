import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
from math import e


class DataUtils:
    """Handles data loading, preprocessing, and splitting"""
    
    def __init__(self):
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
    
    def import_and_split(self, path='CWData.csv', test_ratio=0.2, random_state=42):
        """Load data and split into train/test sets with normalization"""
        df = pd.read_csv(path)
        print(f"Loaded data shape: {df.shape}")
        
        # Separate features and target
        X = df.loc[:, df.columns != 'Index flood']
        y = df['Index flood']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state
        )
        
        # Normalize features
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        X_test_scaled = self.x_scaler.transform(X_test)
        
        # Normalize target
        y_train_scaled = self.y_scaler.fit_transform(y_train.values.reshape(-1, 1)).flatten()
        y_test_scaled = self.y_scaler.transform(y_test.values.reshape(-1, 1)).flatten()
        
        return X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled
    
    def denormalize_predictions(self, y_scaled):
        """Convert normalized predictions back to original scale"""
        return self.y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()


class ErrorMetrics:
    """Handles different error calculations and metrics"""
    
    @staticmethod
    def absolute_error(y_target, y_pred):
        """Calculate absolute error"""
        return np.abs(y_target - y_pred)
    
    @staticmethod
    def mean_absolute_error(y_target, y_pred):
        """Calculate mean absolute error"""
        return np.mean(np.abs(y_target - y_pred))
    
    @staticmethod
    def mean_squared_error(y_target, y_pred):
        """Calculate mean squared error"""
        return np.mean((y_target - y_pred) ** 2)
    
    @staticmethod
    def root_mean_squared_error(y_target, y_pred):
        """Calculate root mean squared error"""
        return np.sqrt(np.mean((y_target - y_pred) ** 2))


class PlottingUtils:
    """Handles all plotting and visualization"""
    
    @staticmethod
    def plot_error_over_epochs(errors, title="Training Error over Epochs"):
        """Plot training error progression"""
        epochs = np.arange(1, len(errors) + 1)
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, errors, marker='o', linestyle='-', linewidth=2)
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_predictions_vs_actual(y_actual, y_pred, title="Predictions vs Actual"):
        """Plot predictions against actual values"""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_actual, y_pred, alpha=0.6)
        plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()