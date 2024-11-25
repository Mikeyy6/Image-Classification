# Import necessary libraries
import torch 
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

# Function that loads the ResNet18 pretrained weights
def load_ResNet18(pretrained = True):

    # Load the pretrained weights using the ResNet18_Weights class
    pretrained_weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    ResNet18 = models.resnet18(weights = pretrained_weights)

    # Remove the last connected layer of ResNet18 
    ResNet18 = nn.Sequential(*list(ResNet18.children())[:-1])
    return ResNet18

# Function that extracts all the features that are needed and ensures the the model is on the correct device
def extract_features(model, dataloader, device, flatten = True):
    
    features_list = []
    labels_list = []

    model.to(device) # Ensure the model is running on the correct device
    model.eval() # Set the model to the evaluation mode

    # Use no_grad() class to avoid gradient computation
    with torch.no_grad():
        
        # For Loop to move all the images to the device
        for images, labels in dataloader:
            obtained_images = images.to(device)

            # Forward pass through the model
            output_results = model(obtained_images)

            # Flatten the results to a 2D Tensor containing the batch size and feature dimension
            if flatten:
                output_results = output_results.view(output_results.size(0), -1) # Flatten to batch_size and 512

            # Store the results in the features and labels list
            features_list.append(output_results.cpu())
            labels_list.append(labels)

    
    # Concatenate all the features and labels into one single Tensor
    all_features = torch.cat(features_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)

    return all_features, all_labels
    
