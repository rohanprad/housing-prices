import numpy as np

np.random.seed(42)

class RidgeRegression:
    
    def __init__(self, alpha = 1):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        self.optimal_theta = None
        
    def fit(self, X_train, y_train):
        X = np.c_[np.ones((len(X_train), 1)), X_train]
        l2_penalty = self.alpha * np.identity(X.shape[1])
        l2_penalty[0][0] = 0
        
        self.optimal_theta = np.linalg.inv(X.T @ X + l2_penalty) @ X.T @ y_train
        #print(self.optimal_theta)
        
        self.intercept_ = self.optimal_theta[0]
        self.coef_ =  self.optimal_theta[1:]
        
        
            
    def predict(self, X_test):
        X = np.c_[np.ones((len(X_test), 1)), X_test]
        y_preds = X @ self.optimal_theta
        return y_preds
    
    def score(self, X_test, y_test):

        if type(X_test) != np.ndarray:
            X_test = X_test.to_numpy()
            y_test = y_test.to_numpy()
            
        y_preds = self.predict(X_test)
        rss = np.sum((y_preds - y_test)**2)
        avg_label = np.mean(y_test)
        tss = np.sum((y_test - avg_label)**2)
        r2 = (tss - rss)/tss
        return r2  