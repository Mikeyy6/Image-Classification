# Import necessary libraries
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Define the MLP Class with adjustable depth and hidden layer size
class MLP(nn.Module):
    
    def __init__(self, input_size: int = 50, hidden_layer_size: int = 512, output_size: int = 10, number_of_layers: int = 2, device=None):
        
        # Initialize the MLP (Multi-layer Perceptron) model
        super(MLP, self).__init__()
        
        # Set device to GPU if available, otherwise use CPU
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define input size, hidden size, output size, and number of layers
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size  
        self.output_size = output_size  
        self.number_of_layers = number_of_layers
        
        # Dynamically create the layers of the network
        self.layers = self.create_layers()
        
        # Move the model to the appropriate device (either CPU or GPU)
        self.to(self.device)

    # Function that dynamically create layers in the neural network
    def create_layers(self):
    
        layers = []
        
        # Add the first (input) layer: input_size -> hidden_layer_size
        layers.append(nn.Linear(self.input_size, self.hidden_layer_size))
        layers.append(nn.ReLU()) 
        
        # Add the hidden layers (varying number of hidden layers depending on number_of_layers)
        for _ in range(self.number_of_layers - 1):  
            layers.append(nn.Linear(self.hidden_layer_size, self.hidden_layer_size))  
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(self.hidden_layer_size))
        
        # Add the output layer: hidden_layer_size to the output_size
        layers.append(nn.Linear(self.hidden_layer_size, self.output_size))
        
        # Return the layers as a sequential model
        return nn.Sequential(*layers)

    # Define the forward pass through the network   
    def forward(self, x):
        
        return self.layers(x)  # Pass input through the layers

    # Train the model using training data
    def fit(self, train_loader: DataLoader, epochs: int):
       
        self.to(self.device)
        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.SGD(params=self.parameters(), lr=0.03, momentum=0.9) 
        
        # Training loop over the specified number of epochs = 20
        for epoch in range(epochs):
            self.train()  
            current_loss = 0.0  
            
            for inputs, labels in train_loader:
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()  

                outputs = self(inputs)
                
                # Compute the loss between the predicted outputs and actual labels
                loss = criterion(outputs, labels)
                
                loss.backward()
                
                optimizer.step()

                current_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {current_loss:.4f}")
        
        return self  # Return the trained model

    # Make Predictions on the test data
    def predict(self, test_features):
        
        self.eval()
    
        test_features = torch.tensor(test_features).float().to(self.device)

        with torch.no_grad():
           
            outputs = self(test_features).argmax(dim=1) 
            
            return outputs.cpu().numpy()
