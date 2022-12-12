import numpy as np

# This distance function uses the Minkowski Distance Formula
# When p = 1, it returns the Manhattan Distance between x1 and x2
# When p = 2, it returns the Euclidean Distance between x1 and x2
def calculate_distance(x1, x2, p = 2):
    return (np.sum(np.abs(x1 - x2)**p))**(1/p)

class KNNRegressor:
    
    def __init__(self, n_neighbors = 3, distance_type = 2, weighted = False):
        self.n_neighbors = n_neighbors
        self.distance_type = distance_type
        self.weighted = weighted 
        print(f'KNNRegressor(n_neighbors={n_neighbors}, distance_type = {distance_type}, weighted = {weighted})')
    
    def fit(self, X_train, y_train):
        if type(X_train) != np.ndarray:
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy()
        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test):
        if type(X_test) != np.ndarray:
            X_test = X_test.to_numpy()
        y_preds = [self._get_prediction(x) for x in X_test]
        return np.array(y_preds)
        
    def _get_prediction(self, x):
        distances = [calculate_distance(x, t, self.distance_type) for t in self.X_train]

        # Getting the indices of k mininum distances (nearest neighbors)
        k_min_indices = np.argsort(distances)[:self.n_neighbors]
                
        # Getting the K nearest labels from the training labels
        k_nearest_labels = [self.y_train[l] for l in k_min_indices]
                
        # Weighted K Nearest Neighbors Regression
        if self.weighted:
            weights = np.arange(self.n_neighbors, 0, -1)
            weights = weights/np.sum(weights)
            return np.dot(weights, k_nearest_labels)
        
        # Unweighted: Assign the mean of the nearest neighbor labels as the label for test sample
        return np.mean(k_nearest_labels)

    
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
    