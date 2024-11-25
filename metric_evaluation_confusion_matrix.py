# Import necessary libraries
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Classification Report Evaluation Function that will print the classification report
def classification_report_evaluation(y_test, y_prediction, model_name):
    
    # Print the classification report
    print("\nClassification Report:\n", classification_report(y_test, y_prediction))

    # Compute and Display the Confusion Matrix
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_prediction))

# Metric Evaluation function that will evaluate the model based on the accuracy, precision, recall and F1-Score
def metric_evaluation(y_test, y_prediction, model_name):
    
    # Evaluate the classifier using accuracy, precision, recall and F1-Score
    accuracy = accuracy_score(y_test, y_prediction)
    precision = precision_score(y_test, y_prediction, average='macro')
    recall = recall_score(y_test, y_prediction, average='macro')
    f1 = f1_score(y_test, y_prediction, average='macro')

    print(f"\n{model_name} Metrics:")
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}\n")

    return accuracy, precision, recall, f1

# Function that will plot the confusion matrix for each case and save it as pdf in the desired path
def confusion_matrix_plot(y_test, y_prediction, class_names, model_name, save_directory = './Confusion_Matrix'):
    
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_directory), exist_ok=True)

    # Calculate the Confusion Matrix
    matrix = confusion_matrix(y_test, y_prediction)

    # Set up the plot using matplotlib.pyplot
    plt.figure(figsize=(10,7))

    # Adjust the heatmap
    sns.heatmap(matrix, annot=True, cmap='coolwarm', 
                xticklabels=class_names, yticklabels=class_names, 
                cbar=True, annot_kws={"size": 12}, linewidths=0.5, linecolor='black')
    
    # Set Axis Labels
    plt.xlabel('Predicted Label', fontsize = 12)
    plt.ylabel('True Label', fontsize = 12)
    plt.title(f'Confusion Matrix - {model_name}', fontsize = 14)

    # Save plot as a pdf file
    plt.savefig(save_directory, format='pdf')
    print(f"Confusion Matrix saved as: {save_directory}")
    plt.close()