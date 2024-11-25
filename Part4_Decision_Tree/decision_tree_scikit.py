# Import necessary libraries
from sklearn.tree import DecisionTreeClassifier

def Scikit_Decision_Tree(x_train, y_train, x_test):

    # Initialize the Decision Tree Classifier
    decision_tree_classifier = DecisionTreeClassifier(random_state=42)
    
    # Train the classifier
    decision_tree_classifier.fit(x_train, y_train)
    
    # Make predictions on the test data
    predictions = decision_tree_classifier.predict(x_test)
    
    return predictions