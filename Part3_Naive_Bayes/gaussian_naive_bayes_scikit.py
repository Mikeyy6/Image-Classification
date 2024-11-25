# Import necessary libraries
# We are using the Gaussian Naive Bayes Classifier for training
from sklearn.naive_bayes import GaussianNB

def Scikit_Gaussian_Naive_Bayes(x_train, y_train, x_test):
    
    # Initialize the Guassian Naive Bayes Classifier
    gaussian_naive_bayes_model = GaussianNB()
    
    # Train the model using the GaussianNB Classifier
    gaussian_naive_bayes_model.fit(x_train, y_train)
    
    # Make the predictiosn on the test dataset
    y_prediction = gaussian_naive_bayes_model.predict(x_test)

    return y_prediction