# Import necessary libraries
import numpy as np

class Gaussian_Naive_Bayes_Algorithm:
    def __init__(self):
        # Define members to store the model parameters needed for this algorithm
        self.class_probabilty = None # Probability
        self.means = None # Mean of the features
        self.variance = None # Variance of the features
        self.classes = None # Class Labels 

    
    # probability_calculation Function that will calcualte the
    # probability as well as the mean and variance for each feature
    def probability_calculation(self, x_train, y_train):
        
        # Define x and y are numpy arrays 
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        # Define Number of samples and Number of Number of features
        number_of_samples, number_of_features = x_train.shape
        self.classes = np.unique(y_train)

        # Initialize the probability, mean and variance
        self.class_probabilty = np.zeros(len(self.classes))
        self.means = np.zeros((len(self.classes), number_of_features))
        self.variance = np.zeros((len(self.classes), number_of_features))

        # Calculate the probability, mean and vairance using a for loop that will compute
        # the probability and a second for loop to compute the mean and variance (nested for loops)
        
        for counter in self.classes:  # Outer loop over classes
            
            # Calculate the class probability (prior P(c))
            class_count = np.sum(y_train == counter)
            self.class_probabilty[counter] = class_count / number_of_samples

            # Initialize lists to store the means and variances for each feature of class `counter`
            x_means = []
            x_variance = []

            # Inner loop over features (assuming x_train is 2D: [n_samples, n_features])
            for index in range(x_train.shape[1]):  # x_train.shape[1] gives the number of features
                
                # Select the data for the current class and feature
                x_data = x_train[y_train == counter, index]

                # Calculate the mean and variance for this feature of class `counter`
                mean = np.mean(x_data)
                variance = np.var(x_data) + 1e-6  # Adding small epsilon to avoid division by zero

                # Append results to the respective lists
                x_means.append(mean)
                x_variance.append(variance)

            # Store the mean and variance for the current class
            self.means[counter] = np.array(x_means)
            self.variance[counter] = np.array(x_variance)


    # Function that calculates and returns the density function based on the mean and variance
    def density_function(self, x, mean, variance):
        
        epsilon = 1e-6 # Avoid division by 0
        density_function = (1.0 / np.sqrt(2* np.pi * variance * epsilon)) * np.exp(-(x - mean) ** 2 / (2 * variance + epsilon))

        return density_function
    

    # Predict function to train the model
    def predict(self, x_test):
        
        x_test = np.array(x_test)

        # Calculate the maximum posterior
        y_prediction = []

        for x_index in x_test:

            # Initialize the log posteriors
            log_posteriors = []

            for index, class_label in enumerate(self.classes):

                # Start with the log of prior
                log_posterior = np.log(self.class_probabilty[index])

                # Add the log of likelihood
                likelihood = np.sum(np.log(self.density_function(x_index, self.means[index], self.variance[index])))

                # Compute the total log 
                log_posteriors.append(log_posterior + likelihood)

            # Append the class with the highest log
            y_prediction.append(self.classes[np.argmax(log_posteriors)])

        return np.array(y_prediction)        
    
