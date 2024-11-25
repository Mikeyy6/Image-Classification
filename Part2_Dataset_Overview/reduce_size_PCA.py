# Import necessary libraries
import numpy as np
from sklearn.decomposition import PCA
import os

# Function that will apply the PCA to reduce the size of 
# a feature dimension from 512 x 1 to 50 x 1

def pca(train_features, test_features, number_components = 50):
    
    # Fit the PCA model on the training set
    Pca = PCA(n_components=number_components)

    # Transform both Training and Test features using teh same PCA
    train_features_pca = Pca.fit_transform(train_features)
    test_features_pca = Pca.transform(test_features)

    return train_features_pca, test_features_pca

# Function that will save the new featues / labels as npy files
def save_features(train_features, train_labels, test_features, test_labels, save_directory = './Features_PCA'):
    
    # Create save directory if it doesn't exist
    os.makedirs(save_directory, exist_ok=True)

    # Define file paths
    train_features_path = os.path.join(save_directory, 'Train_Features_PCA.npy')
    train_labels_path = os.path.join(save_directory, 'Train_Labels.npy')
    test_features_path = os.path.join(save_directory, 'Test_Features_PCA.npy')
    test_labels_path = os.path.join(save_directory, 'Test_Labels.npy')

    # Save the features and labels
    np.save(train_features_path, train_features)
    np.save(train_labels_path, train_labels)
    np.save(test_features_path, test_features)
    np.save(test_labels_path, test_labels)

    print(f"\nFeatures and labels saved to {save_directory}")