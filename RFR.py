import numpy as np

np.random.seed(42)

class Node:
    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        
    def is_leaf(self):
        if self.value is not None:
            return True
        else:
            return False

def get_mse(y):
    y_pred = np.mean(y)
    return np.mean((y-y_pred)**2)



class DecisionTreeRegressor:
    def __init__(self, min_samples_split = 2, max_depth = None, max_features = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.root = None
        
    def fit(self, X_train, y_train):
        if type(X_train) != np.ndarray:
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy()
        if self.max_features is None:
            self.max_features = X_train.shape[1]
        else:
            self.max_features = min(self.max_features, X_train.shape[1])
        
        self.root = self._generate_tree(X_train, y_train)
        
    def predict(self, X_test):
        if type(X_test) != np.ndarray:
            X_test = X_test.to_numpy()
        y_preds = [self._get_prediction(x, self.root) for x in X_test]
        return y_preds
    
    def _generate_tree(self, X_train, y_train, depth = 0):
        n_samples, n_features = X_train.shape
        n_labels = len(np.unique(y_train))
        
        if n_samples > self.min_samples_split and depth < self.max_depth and n_labels > 1 :
            feature_indices = np.random.choice(n_features, self.max_features, replace = False)
            
            
            best_feature, best_threshold = self._get_best_split(X_train, y_train, feature_indices)
            
            l_indices, r_indices = self._split(X_train[:, best_feature], best_threshold)
            left = self._generate_tree(X_train[l_indices, :], y_train[l_indices], depth + 1)
            right = self._generate_tree(X_train[r_indices, :], y_train[r_indices], depth + 1)
            
            return Node(best_feature, best_threshold, left, right)
            
             
        # Assign the average of the labels in the branch
        if len(y_train) != 0:
            leaf_value = np.mean(y_train)
        else:
            leaf_value = 0
        return Node(value = leaf_value)
    
    def _get_best_split(self, X_train, y_train, feature_indices):
        best_info_gain = float('-inf')
        split_index,  split_threshold = None, None
        for feature_index in feature_indices:
            #print(feature_index)
            X_column = X_train[:, feature_index]
            possible_thresholds = np.unique(X_column)
            for threshold in possible_thresholds:
                info_gain = self._get_information_gain(X_column, y_train, threshold)
                
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    split_index = feature_index
                    split_threshold = threshold
                    
        return split_index, split_threshold
    
    def _get_information_gain(self, X_column, y_train, split_threshold):
        parent_mse = get_mse(y_train)
        
        l_indices, r_indices = self._split(X_column, split_threshold)
        
        if len(l_indices) == 0 or len(r_indices) == 0:
            return 0
        
        n = len(y_train)
        weight_l = len(l_indices)/n
        weight_r = len(r_indices)/n
        
        l_mse = get_mse(y_train[l_indices])
        r_mse = get_mse(y_train[r_indices])
        
        child_mse = (weight_l * l_mse) + (weight_r * r_mse)
        
        info_gain = parent_mse - child_mse
        
        return info_gain
    
    def _split(self, X_column, split_threshold):
        l_indices = np.argwhere(X_column <= split_threshold).flatten()
        r_indices = np.argwhere(X_column > split_threshold).flatten()
        return l_indices, r_indices
    
    def _get_prediction(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._get_prediction(x, node.left)
        return self._get_prediction(x, node.right)
    
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

##---------------------------Random Forest Implementation------------------------##

def get_bootstrapped_dataset(X, y):
    n_samples = X.shape[0]
    indices = np.random.choice(n_samples, n_samples, replace = True)
    return X[indices], y[indices]
    

class RandomForestRegressor:
    def __init__(self, n_estimators = 10, min_samples_split = 2, max_depth = 100, max_features = None):
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        
    def fit(self, X_train, y_train):
        if type(X_train) != np.ndarray:
            X_train = X_train.to_numpy()
            y_train = y_train.to_numpy()
        self.trees = []
        for i in range(self.n_estimators):
            tree = DecisionTreeRegressor(min_samples_split = self.min_samples_split,
                                        max_depth = self.max_depth,
                                        max_features=self.max_features)
            X_bootstrapped, y_bootstrapped = get_bootstrapped_dataset(X_train, y_train)
            tree.fit(X_bootstrapped, y_bootstrapped)
            self.trees.append(tree)
            
    def predict(self, X_test):
        if type(X_test) != np.ndarray:
            X_test = X_test.to_numpy()
        tree_preds = np.array([tree.predict(X_test) for tree in self.trees])
        #print(tree_preds.shape)
        y_preds = np.array([np.mean(tree_pred) for tree_pred in np.transpose(tree_preds)])
        #print(y_preds.shape)
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