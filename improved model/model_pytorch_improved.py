import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ImprovedCNN(nn.Module):
    """
    Improved Convolutional Neural Network model for MNIST classification.
    Includes batch normalization, dropout, and a more sophisticated architecture.
    """
    def __init__(self, dropout_rate=0.25):
        super(ImprovedCNN, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Third convolutional block
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate the size after 3 pooling layers (28x28 -> 14x14 -> 7x7 -> 3x3)
        self.flatten_size = 128 * 3 * 3
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 10)
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for faster and more stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
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
    model = ImprovedCNN()
    
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
    
    # Count parameters
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {pytorch_total_params:,}") 