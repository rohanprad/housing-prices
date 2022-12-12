import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class OnlineLinearRegression:
    def __init__(self, n_features, scale_index = -1):
        self.X_train = []
        self.X_train_scaled = None
        self.losses = []
        self.scaler = StandardScaler()
        self.weights = np.zeros(n_features)
        self.cumulative_loss_ = 0
        self.scale_index = scale_index
        
    def _custom_transform(self, X):
        X_copy = np.copy(X)
        X_copy[:, :self.scale_index] = self.scaler.transform(X_copy[:, :self.scale_index])
        return X_copy
              
        
    def predict(self, x):
        # Append new sample to train set
        self.X_train.append(x)
        
        # Perform Feature Scaling
        if self.scale_index != -1:
            self.scaler.fit(pd.DataFrame(self.X_train).iloc[:, :self.scale_index])
            self.X_train_scaled = self._custom_transform(self.X_train)
        else:
            self.scaler.fit(pd.DataFrame(self.X_train))
            self.X_train_scaled = self.scaler.transform(self.X_train)
 
        # Make Prediction
        return np.dot(self.weights, self.X_train_scaled[-1])
    
        
    def update_weights(self, y, y_pred):
        self.weights =  self.weights - 0.01  * ((y_pred - y) * self.X_train_scaled[-1])
        loss = (y_pred - y)**2
        self.losses.append(loss) 
        self.cumulative_loss_ = np.sum(self.losses)