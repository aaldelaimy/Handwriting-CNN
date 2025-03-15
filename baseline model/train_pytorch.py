import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from model_pytorch import CNN

def load_mnist_pytorch(data_dir='data'):
    """Load MNIST dataset and convert to PyTorch tensors."""
    from mnist_loader import load_mnist
    
    # Load data using our existing loader
    (train_images, train_labels, _), (test_images, test_labels, _) = load_mnist(data_dir)
    
    # Convert to PyTorch tensors
    train_images_torch = torch.FloatTensor(train_images)
    train_labels_torch = torch.LongTensor(train_labels)
    test_images_torch = torch.FloatTensor(test_images)
    test_labels_torch = torch.LongTensor(test_labels)
    
    return (train_images_torch, train_labels_torch), (test_images_torch, test_labels_torch)

def load_small_mnist_pytorch(samples=5000, random_seed=42):
    """Create and load a smaller MNIST dataset for faster training."""
    from create_small_dataset import create_small_mnist
    
    # Create small dataset if it doesn't exist
    if not os.path.exists('small_data/train_images.npy'):
        create_small_mnist(samples=samples, random_seed=random_seed)
    
    # Load the small dataset
    train_images = np.load('small_data/train_images.npy')
    train_labels = np.load('small_data/train_labels.npy')
    test_images = np.load('small_data/test_images.npy')
    test_labels = np.load('small_data/test_labels.npy')
    
    # Convert to PyTorch tensors
    train_images_torch = torch.FloatTensor(train_images)
    train_labels_torch = torch.LongTensor(train_labels)
    test_images_torch = torch.FloatTensor(test_images)
    test_labels_torch = torch.LongTensor(test_labels)
    
    return (train_images_torch, train_labels_torch), (test_images_torch, test_labels_torch)

def train(args):
    """Train the CNN model using PyTorch."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    if args.use_small_dataset:
        (train_images, train_labels), (test_images, test_labels) = load_small_mnist_pytorch(samples=args.n_samples)
    else:
        (train_images, train_labels), (test_images, test_labels) = load_mnist_pytorch()
    
    print(f"Train images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")
    
    # Create data loaders
    train_dataset = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model and move to device
    model = CNN().to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # Initialize stats
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_test_accuracy = 0
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        # Progress bar for training
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': train_loss / (progress_bar.n + 1),
                'acc': 100. * correct / total
            })
        
        # Calculate training accuracy for this epoch
        train_accuracy = 100. * correct / total
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Statistics
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate test accuracy for this epoch
        test_accuracy = 100. * correct / total
        test_loss = test_loss / len(test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Print statistics
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.2f}%")
        print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.2f}%")
        
        # Save best model
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            model.save_weights(os.path.join(args.output_dir, "best_model.pt"))
            print(f"New best model saved with test accuracy: {test_accuracy:.2f}%")
    
    # Save final model
    model.save_weights(os.path.join(args.output_dir, "final_model.pt"))
    
    # Plot training curves
    plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies, args.output_dir)
    
    # Visualize some predictions
    visualize_predictions(model, test_images, test_labels, device, args.output_dir)
    
    print(f"Training completed. Best test accuracy: {best_test_accuracy:.2f}%")

def plot_training_curves(train_losses, test_losses, train_accuracies, test_accuracies, output_dir):
    """Plot training and testing curves."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(test_accuracies, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'))
    plt.close()

def visualize_predictions(model, images, labels, device, output_dir, num_samples=10):
    """Visualize model predictions."""
    model.eval()
    
    # Randomly select samples
    indices = torch.randperm(len(images))[:num_samples]
    sample_images = images[indices]
    sample_labels = labels[indices]
    
    # Make predictions
    sample_images = sample_images.to(device)
    with torch.no_grad():
        predictions = model.predict(sample_images)
    predicted_labels = torch.argmax(predictions, dim=1).cpu()
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Plot image
        image = sample_images[i, 0].cpu().numpy()
        axes[i].imshow(image, cmap='gray')
        
        # Set title (green if correct, red if wrong)
        title_color = 'green' if predicted_labels[i] == sample_labels[i] else 'red'
        axes[i].set_title(f"True: {sample_labels[i].item()}\nPred: {predicted_labels[i].item()}", 
                         color=title_color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train CNN for MNIST digit classification using PyTorch')
    
    # Dataset parameters
    parser.add_argument('--use_small_dataset', action='store_true', 
                        help='use a smaller subset of MNIST for faster training')
    parser.add_argument('--n_samples', type=int, default=5000, 
                        help='number of training samples to use if using small dataset')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, 
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, 
                        help='initial learning rate')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output_pytorch', 
                        help='directory to save results')
    
    args = parser.parse_args()
    print(args)
    
    # Start training
    train(args)

if __name__ == "__main__":
    main() 