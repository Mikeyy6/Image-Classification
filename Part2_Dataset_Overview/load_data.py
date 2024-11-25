# Import necessary libraries
import torchvision
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import numpy as np

# Define a convert function that will resize and normalize the images from the dataset
# The function returns the transformed images that were inputted
def normalize_dataset(size_of_images = (224, 224)):
    return transforms.Compose([
        transforms.Resize(size_of_images), 
        transforms.ToTensor(), 
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# The return line resizes the images from the dataset to the desired size which is 224 x 224 x 3
# then converts the images into a Tensor and normalizes the images based on RESNET-18 

# Define function that uses the Subset import to return the samples that are used from each class
def sample_count(dataset_count, samples_in_class, number_of_classes=10):
    counter = {index_ctr: 0 for index_ctr in range (number_of_classes)} # Track Sample Counts for each class
    array_indexes = []

    for index, (_, class_label) in enumerate (dataset_count):

        if counter[class_label] < samples_in_class:
            array_indexes.append(index)
            counter[class_label] += 1

        # Stop once obtained all the desired number of data
        if all(obtained_samples == samples_in_class for obtained_samples in counter.values()):
            break
        
    return Subset(dataset_count, array_indexes)

# Define function that will load the resized images from the CIFAR10 dataset
def load_data(samples_in_class = 500, batch_size = 32, size_of_images = (224, 224), number_of_classes = 10):

    # Load the CIFAR10 Dataset
    dataset_name = torchvision.datasets.CIFAR10

    # Apply the normalize function to obtain the data in the correct format
    converted_dataset = normalize_dataset(size_of_images=size_of_images)

    # Load and Train the datasets
    train_data = dataset_name(root='./CIFAR10_Data', train = True, download = True, transform = converted_dataset)
    test_data = dataset_name(root='./CIFAR10_Data', train = False, download = True, transform = converted_dataset)

    # Create subsets for both training and testing data
    train_subset = sample_count(train_data, samples_in_class, number_of_classes = number_of_classes)
    test_subset = sample_count(test_data, 100, number_of_classes = number_of_classes) # 100 samples for testing 

    # Create DataLoaders
    train_loader = DataLoader(train_subset,batch_size = batch_size, shuffle = True, num_workers=2)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle = False, num_workers=2)

    return train_loader, test_loader