# Import necessary libraries and functions
import torch
import numpy as np
import time
import os, random
import joblib
import torch.nn as nn
from tabulate import tabulate

# Import Functions from other directories
from Part2_Dataset_Overview.load_data import load_data
from Part2_Dataset_Overview.extract_data import load_ResNet18, extract_features
from Part2_Dataset_Overview.reduce_size_PCA import pca, save_features
from Part3_Naive_Bayes.gaussian_naive_bayes import Gaussian_Naive_Bayes_Algorithm
from Part3_Naive_Bayes.gaussian_naive_bayes_scikit import Scikit_Gaussian_Naive_Bayes
from Part4_Decision_Tree.decision_tree_scikit import Scikit_Decision_Tree  
from Part4_Decision_Tree.decision_tree import DecisionTree
from Part5_Multi_Layer_Perceptron.multi_layer_perceptron import MLP
from Part6_Convolutional_Neural_Network.CNN import CNN_VGG11_Model
from Part6_Convolutional_Neural_Network.CNN import evaluate_model, train_model
from metric_evaluation_confusion_matrix import classification_report_evaluation, confusion_matrix_plot, metric_evaluation
from torch.utils.data import DataLoader, TensorDataset

# Main Function
def main():

    # Set random seeds for reproducibility
    random.seed(42)  # Python's random module
    np.random.seed(42)  # NumPy random
    torch.manual_seed(42)  # PyTorch random seed

    # Check if the device is GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the Data
    x_load, y_load = load_data(samples_in_class=500, batch_size=32)

    # Load Pretrained Data from ResNet18
    ResNet18 = load_ResNet18()
    ResNet18.to(device)

    # Extract Features
    print("\nExtracting Features")
    x_train, y_train = extract_features(ResNet18, x_load, device)
    x_test, y_test = extract_features(ResNet18, y_load, device)

    print(f"\nTrain Features Shape: {x_train.shape}")  # Should print (5000, 512)
    print(f"Test Features Shape: {x_test.shape}")  # Should print (1000, 512)

    # Convert Tensors to Numpy arrays
    x_train = x_train.cpu().numpy()
    x_test = x_test.cpu().numpy()

    # Apply the PCA to reduce the dimensions
    print("\nReducing Dimensions for PCA:")
    x_train_pca, x_test_pca = pca(x_train, x_test, number_components=50)

    print(f"\nTraining Features after reducing dimensions: {x_train_pca.shape}")  # should print (5000, 50)
    print(f"Test Features after reducing dimensions: {x_test_pca.shape}")  # Should print (1000, 50)

    # Save Features and Labels (X and Y)
    save_features(x_train_pca, y_train, x_test_pca, y_test)

    print("Feature Extraction and PCA have been saved!")
    # Load CIFAR-10 data (raw images)
    train_loader, test_loader = load_data(samples_in_class=500, batch_size=64, size_of_images=(32, 32))    

# ====================================================================================================================================
# ========================================= Convolutional Neural Network =============================================================
    # List of kernel sizes to experiment with
    kernel_sizes = [3, 5, 7]

    # Iterate through the kernel sizes
    for kernel_size in kernel_sizes:
        print(f"\nInitializing CNN VGG11 Model with kernel size: {kernel_size}")

        # Dynamically choose model based on kernel size
        cnn_model = CNN_VGG11_Model(num_classes=10, kernel_size=kernel_size).to(device)

        # Define criterion and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(cnn_model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # Train the model
        print(f"\nTraining CNN VGG11 Model with kernel size: {kernel_size}")
        train_model(cnn_model, train_loader, criterion, optimizer, num_epochs=20)

        # Evaluate the model
        print(f"\nEvaluating CNN VGG11 Model with kernel size: {kernel_size}")
        evaluate_model(cnn_model, test_loader)

        # Save the trained CNN model
        model_save_path = f'./Pretrained_Models/Convolutional_Neural_Network/CNN_VGG11_Model_Kernel_Size_{kernel_size}.pth'
        torch.save(cnn_model.state_dict(), model_save_path)
        print(f"CNN VGG11 Model with kernel size {kernel_size} has been saved to {model_save_path}")

        # Set model to evaluation mode
        cnn_model.eval()

        # Collect predictions and true labels for confusion matrix
        cnn_predictions = []
        y_test = []

        # Disable gradient computation for evaluation
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.cuda(), labels.cuda()

                # Forward pass
                outputs = cnn_model(inputs)
                _, predicted = torch.max(outputs, 1)

                # Collect true labels and predicted labels
                cnn_predictions.extend(predicted.cpu().numpy())
                y_test.extend(labels.cpu().numpy())

        # Generate confusion matrix plot
        confusion_matrix_plot(y_test, cnn_predictions, list(range(10)),
                            f"Convolutional Neural Network (CNN) with kernel size {kernel_size}",
                            save_directory=f'./Confusion_Matrix/Convolutional_Neural_Network/CNN_Confusion_Matrix_Kernel_Size_{kernel_size}.pdf')

# =======================================================================================================================================
# ======================================== Gaussian Naive Bayes =========================================================================
    
    # Save/load Gaussian Naive Bayes model
    custom_gaussian_naive_bayes = './Pretrained_Models/Gaussian_Naive_Bayes/Custom_Gaussian_Naive_Bayes.pth'
    
    if os.path.exists(custom_gaussian_naive_bayes):
        
        print("\nLoading pre-trained Gaussian Naive Bayes model...")
        gaussian_naive_bayes_custom = torch.load(custom_gaussian_naive_bayes)
        
        # Load the predictions if the model is pre-trained
        predictions_naive_bayes_custom = gaussian_naive_bayes_custom.predict(x_test_pca)  # Ensure predictions are loaded
    
    else:
        
        # Train, Evaluate, and Plot Confusion Matrix for Custom Gaussian Naive Bayes
        gaussian_naive_bayes_custom = Gaussian_Naive_Bayes_Algorithm()
        gaussian_naive_bayes_custom.probability_calculation(x_train_pca, y_train)  # Train the custom model
        predictions_naive_bayes_custom = gaussian_naive_bayes_custom.predict(x_test_pca)  # Predictions on the test data
        torch.save(gaussian_naive_bayes_custom, custom_gaussian_naive_bayes)  # Save the model for future use

    # Now, call the metric evaluation function (this will work regardless of whether the model was pre-trained or newly trained)
    metric_evaluation(y_test, predictions_naive_bayes_custom, "Custom Gaussian Naive Bayes")
    classification_report_evaluation(y_test, predictions_naive_bayes_custom, "Custom Gaussian Naive Bayes")
    confusion_matrix_plot(y_test, predictions_naive_bayes_custom, list(range(10)), "Custom Gaussian Naive Bayes", 
                        save_directory='./Confusion_Matrix/Naive_Bayes/Custom_Naive_Bayes.pdf')


    # Train, Evaluate and Plot Confusion Matrix for Scikit-Learn Gaussian Naive Bayes

    scikit_predictions = Scikit_Gaussian_Naive_Bayes(x_train_pca, y_train, x_test_pca)

    metric_evaluation(y_test, scikit_predictions, "Scikit Learn Gaussian Naive Bayes")
    classification_report_evaluation(y_test, scikit_predictions, "Scikit-Learn Gaussian Naive Bayes")
    confusion_matrix_plot(y_test, scikit_predictions, list(range(10)), "Scikit-Learn Gaussian Naive Bayes", 
                          save_directory='./Confusion_Matrix/Naive_Bayes/Scikit_Naive_Bayes.pdf')

# =======================================================================================================================================
# =========================================== Decision Tree Classifier ===============================================================
    
    # Train, Evaluate, and Plot Confusion Matrix for Custom Decision Tree 

    # List of depths to evaluate
    depths = [1, 5, 10, 50]  # List of depths to evaluate

    # Loop through each depth and train, evaluate, and save/load the model
    for depth in depths:
        print(f"\nEvaluating Custom Decision Tree with depth of {depth}:")

        # Define filename based on depth
        model_filename = f'./Pretrained_Models/Decision_Tree/Custom_Decision_Tree_Depth_{depth}.pkl'

        # Check if the model is already trained and saved
        try:
            # Try to load the pre-trained custom decision tree model for the specific depth
            decision_tree_model = joblib.load(model_filename)
            print(f"Loaded pre-trained Custom Decision Tree model with depth {depth} from {model_filename}.")
        except FileNotFoundError:
            # If the model is not found, train a new one and save it
            print(f"Training Custom Decision Tree model with depth {depth}.")
            decision_tree_model = DecisionTree(max_depth=depth)
            decision_tree_model.fit(x_train_pca, y_train)
            
            # Save the trained custom model
            joblib.dump(decision_tree_model, model_filename)
            print(f"Custom model saved as {model_filename}.")

        # Make predictions using the (either pre-trained or newly trained) Custom Decision Tree model
        decision_tree_predictions = decision_tree_model.predict(x_test_pca)

        # Evaluate the model (metrics, classification report, confusion matrix, etc.)
        metric_evaluation(y_test, decision_tree_predictions, f"Custom Decision Tree Classifier with depth {depth}")
        classification_report_evaluation(y_test, decision_tree_predictions, f"Custom Decision Tree Classifier with depth {depth}")
        confusion_matrix_plot(y_test, decision_tree_predictions, list(range(10)), 
                              f"Custom Decision Tree Classifier (max_depth={depth})", 
                              save_directory=f'./Confusion_Matrix/Decision_Tree/Custom_Decision_Tree_Depth_{depth}.pdf')

    # Train, Evaluate, and Plot Confusion Matrix for Scikit Decision Tree 
    
    # Train the Scikit-Learn Decision Tree model
    scikit_decision_tree_predictions = Scikit_Decision_Tree(x_train_pca, y_train, x_test_pca)
    
    # Evaluate the Scikit Decision Tree model with classification report
    metric_evaluation(y_test, scikit_decision_tree_predictions, "Scikit-Learn Decision Tree Classifier")
    
    # Plot and save the confusion matrix for the Scikit-Learn Decision Tree model
    confusion_matrix_plot(y_test, scikit_decision_tree_predictions, list(range(10)), "Scikit-Learn Decision Tree Classifier", 
                          save_directory='./Confusion_Matrix/Decision_Tree/Scikit_Decision_Tree.pdf')
    
# ====================================================================================================================================
# ============================================= Multi-Layer Perceptron ===============================================================
    # Initialize and Train the MLP
    print("\nTraining Multi-Layer Perceptron (MLP):")
    
    # Wrap the data into DataLoader for batching
    train_data = TensorDataset(torch.tensor(x_train_pca).float(), torch.tensor(y_train).long())
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    # Initialize the MLP model
    mlp_model = MLP(input_size=50, hidden_layer_size=512, output_size=10)  # Adjust architecture as needed

    # Train the MLP model
    print("\nTraining Multi-Layer Perceptron (MLP):")
    mlp_model.fit(train_loader, epochs=20)  # Pass DataLoader and epochs as keyword argument

    # Make predictions using the MLP model
    mlp_predictions = mlp_model.predict(x_test_pca)

    # Evaluate the MLP model
    metric_evaluation(y_test, mlp_predictions, "Multi-Layer Perceptron (MLP)")

    # Plot and save the confusion matrix for MLP
    confusion_matrix_plot(y_test, mlp_predictions, list(range(10)), "Multi-Layer Perceptron (MLP)", 
                            save_directory='./Confusion_Matrix/Multi_Layer_Perceptron/MLP_Confusion_Matrix.pdf')
    
    hidden_layer_sizes = [128, 256, 512]  # Example hidden layer sizes
    depths = [1, 2, 3]  # Example depths (number of hidden layers)
    
    results = []

    # Wrap the data into DataLoader for batching
    train_data = TensorDataset(torch.tensor(x_train_pca).float(), torch.tensor(y_train).long())
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    for depth in depths:
        for hidden_size in hidden_layer_sizes:
            print(f"\nTraining MLP with {depth} hidden layers and {hidden_size} hidden units per layer:")

            # Initialize MLP model with current depth and hidden size
            mlp_model = MLP(input_size=50, hidden_layer_size=hidden_size, output_size=10, number_of_layers=depth, device=device)
            
            # Measure training time
            start_time = time.time()

            # Train the model
            mlp_model.fit(train_loader, epochs=20)

            # Save the trained model for this configuration
            model_filename = f'./Pretrained_Models/Multi_Layer_Perceptron/MLP_Model_Depth_{depth}_HL_{hidden_size}.pth'
            torch.save(mlp_model.state_dict(), model_filename)
            print(f"Model saved as {model_filename}")

            # Measure training time
            training_time = time.time() - start_time
            print(f"\nTraining time: {training_time:.2f} seconds")

            # Make predictions on the test set
            mlp_predictions = mlp_model.predict(x_test_pca)

            # Ensure mlp_predictions is a NumPy array (if it is a tensor, convert it)
            mlp_predictions = mlp_predictions.cpu().numpy() if isinstance(mlp_predictions, torch.Tensor) else mlp_predictions

            # Ensure y_test is also a NumPy array (if it is a tensor, convert it)
            y_test = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

            # Calculate accuracy using NumPy
            accuracy = np.mean(mlp_predictions == y_test)
            print(f"Test accuracy: {accuracy:.4f}")

            # Store results for later analysis
            results.append((depth, hidden_size, accuracy, training_time))

    # Now print results in a tabular format
    header = ["Depth", "Hidden Units", "Test Accuracy", "Training Time (s)"]
    print("\nResults:\n")
    print(tabulate(results, headers=header, floatfmt=(".2f", ".4f", ".2f")))

# =====================================================================================================================================
    
if __name__ == "__main__":
    main()