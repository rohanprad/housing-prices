import numpy as np
np.random.seed(42)

class KernelizedRidgeRegression:
    
    def __init__(self, alpha = 1, kernel = 'linear', gamma = None, degree = 3, coef0 = 1):
        self.alpha = alpha
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.optimal_r = None
        self.gamma = gamma
        
        
    def fit(self, X_train, y_train):
        if self.gamma == None:
            self.gamma = 1.0/X_train.shape[1]
        self.X = X_train
        l2_penalty = self.alpha * np.identity(self.X.shape[0])
        
        self.optimal_r = y_train.T @ np.linalg.inv(self._kernel(self.X, self.X.T) + l2_penalty)
        
        
    def _kernel(self, A, B):
        if self.kernel == 'linear':
            return A @ B
        elif self.kernel == 'polynomial':
            return ((self.gamma * (A @ B)) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            if type(A) != np.ndarray:
                A = A.to_numpy()
            if type(B) != np.ndarray:
                B = B.to_numpy()
            A_norm = np.sum(A ** 2, axis = -1)
            B_norm = np.sum(B ** 2, axis = -1)
            if A.shape[0] != B.shape[0] and A.shape[1] == B.shape[1]:
                 #return pairwise.rbf_kernel(A, B)
                return np.exp(-self.gamma * (A_norm[:, None] + B_norm[None, :] - 2 * np.dot(A,B.T)))
            else:
                 #return pairwise.rbf_kernel(A)
                return np.exp(-self.gamma * (A_norm[None, :] + A_norm[:, None] - 2 * np.dot(A,A.T)))
                   
            
    def predict(self, X_test):
        if self.kernel == 'rbf':
            y_preds = self.optimal_r @ self._kernel(self.X, X_test)
        else:
            y_preds = self.optimal_r @ self._kernel(self.X, X_test.T)
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