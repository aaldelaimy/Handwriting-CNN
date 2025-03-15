import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class CNN(nn.Module):
    """
    Convolutional Neural Network model implemented using PyTorch.
    This maintains the same architecture as our NumPy implementation,
    but leverages PyTorch's optimized tensor operations.
    """
    def __init__(self):
        super(CNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        """Forward pass through the network."""
        # First conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """
        Make predictions for input data.
        Returns class probabilities after softmax.
        """
        # Forward pass
        logits = self.forward(x)
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=1)
        
        return probs
    
    def save_weights(self, filepath):
        """Save model weights to a file."""
        torch.save(self.state_dict(), filepath)
        print(f"Model weights saved to {filepath}")
    
    def load_weights(self, filepath):
        """Load model weights from a file."""
        self.load_state_dict(torch.load(filepath))
        print(f"Model weights loaded from {filepath}")

# Testing (if run as a script)
if __name__ == "__main__":
    # Create a model
    model = CNN()
    
    # Print model architecture
    print(model)
    
    # Forward pass with random data
    x = torch.randn(10, 1, 28, 28)  # Batch of 10 MNIST-like images
    logits = model(x)
    print(f"Output shape: {logits.shape}")
    
    # Test predictions
    probs = model.predict(x)
    print(f"Prediction probabilities shape: {probs.shape}")
    print(f"Probability sums: {probs.sum(dim=1)}")  # Should be close to 1
    
    # Test with optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # Random targets
    targets = torch.randint(0, 10, (10,))
    
    # Compute loss
    loss = criterion(logits, targets)
    print(f"Loss: {loss.item()}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Save and load weights
    model.save_weights("test_weights_pytorch.pt")
    model.load_weights("test_weights_pytorch.pt") 