# Import necessary libraries
import numpy as np

# Gini Impurity Function
def gini_impurity(y):
    class_counts = np.bincount(y)
    total_samples = len(y)
    impurity = 1.0
    for count in class_counts:
        probability = count / total_samples
        impurity -= probability ** 2
    return impurity

# Split Data Based on Feature and Threshold (returning indices instead of data arrays)
def split_data(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return left_mask, right_mask

# Find the Best Split Using Gini Impurity
def best_split(X, y):
    best_gini = float('inf')
    best_feature = None
    best_threshold = None
    best_left_mask = None
    best_right_mask = None
    
    number_of_samples, number_of_features = X.shape
    for feature_index in range(number_of_features):
        thresholds = np.unique(X[:, feature_index])
        
        for threshold in thresholds:
            left_mask, right_mask = split_data(X, y, feature_index, threshold)
            if np.any(left_mask) and np.any(right_mask):  # Skip invalid splits
                left_y = y[left_mask]
                right_y = y[right_mask]
                left_weight = len(left_y) / number_of_samples
                right_weight = len(right_y) / number_of_samples
                gini_left = gini_impurity(left_y)
                gini_right = gini_impurity(right_y)
                gini = left_weight * gini_left + right_weight * gini_right
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask
    
    return best_feature, best_threshold, best_left_mask, best_right_mask

# Decision Tree Node Class
class DecisionTreeNode:
    def __init__(self, gini, number_of_samples, number_of_samples_per_class, predicted_class):
        self.gini = gini
        self.number_of_samples = number_of_samples
        self.number_of_samples_per_class = number_of_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None

# Decision Tree Classifier
class DecisionTree:
    def __init__(self, max_depth=50):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        number_of_samples, number_of_features = X.shape
        number_of_classes = len(np.unique(y))
        gini = gini_impurity(y)
        
        # Stopping criteria: max depth, perfect classification, or too few samples
        if depth >= self.max_depth or gini == 0 or number_of_samples <= 1:
            predicted_class = np.bincount(y).argmax()
            return DecisionTreeNode(gini=gini, number_of_samples=number_of_samples,
                                    number_of_samples_per_class=np.bincount(y),
                                    predicted_class=predicted_class)
        
        # Find the best split
        feature_index, threshold, left_mask, right_mask = best_split(X, y)
        
        if feature_index is None:  # If no valid split was found, make a leaf node
            predicted_class = np.bincount(y).argmax()
            return DecisionTreeNode(gini=gini, number_of_samples=number_of_samples,
                                    number_of_samples_per_class=np.bincount(y),
                                    predicted_class=predicted_class)
        
        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        node = DecisionTreeNode(gini=gini, number_of_samples=number_of_samples,
                                number_of_samples_per_class=np.bincount(y),
                                predicted_class=np.bincount(y).argmax())
        node.feature_index = feature_index
        node.threshold = threshold
        node.left = left_node
        node.right = right_node
        
        return node

    def predict(self, X):
        return [self._predict_instance(x) for x in X]

    def _predict_instance(self, x):
        node = self.root
        while node.left:  # Traverse the tree
            if x[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class
