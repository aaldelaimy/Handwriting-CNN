import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from model_pytorch_improved import ImprovedCNN
from mnist_loader import load_mnist

def load_mnist_pytorch(data_dir='data'):
    """Load MNIST dataset and convert to PyTorch tensors with data augmentation."""
    # Load data using our loader
    (train_images, train_labels, _), (test_images, test_labels, _) = load_mnist(data_dir)
    
    # Convert to PyTorch tensors
    train_images_torch = torch.FloatTensor(train_images)
    train_labels_torch = torch.LongTensor(train_labels)
    test_images_torch = torch.FloatTensor(test_images)
    test_labels_torch = torch.LongTensor(test_labels)
    
    return (train_images_torch, train_labels_torch), (test_images_torch, test_labels_torch)

class AugmentedMNIST(torch.utils.data.Dataset):
    """MNIST dataset with data augmentation."""
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels
        self.augment = augment
        
        # Augmentation transforms
        self.transform = transforms.Compose([
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        # Apply augmentation only to training set
        if self.augment:
            image = self.transform(image)
            
        return image, label

def train(args):
    """Train the CNN model using advanced techniques."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load MNIST dataset
    print("Loading MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = load_mnist_pytorch()
    
    print(f"Train images shape: {train_images.shape}")
    print(f"Test images shape: {test_images.shape}")
    
    # Split training data into train and validation sets
    train_dataset = AugmentedMNIST(train_images, train_labels, augment=True)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # No augmentation for validation and test
    val_dataset.dataset.augment = False
    test_dataset = AugmentedMNIST(test_images, test_labels, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model and move to device
    model = ImprovedCNN(dropout_rate=args.dropout_rate).to(device)
    
    # Print model summary
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler - reduce on plateau
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.1, 
        patience=3,
        verbose=True
    )
    
    # Initialize stats
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    best_val_accuracy = 0
    epochs_without_improvement = 0
    
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
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
        
        # Evaluate on validation set
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Print statistics
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"Train loss: {train_loss:.4f}, Train accuracy: {val_accuracy:.2f}%")
        print(f"Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.2f}%")
        
        # Step the learning rate scheduler based on validation accuracy
        scheduler.step(val_accuracy)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model.save_weights(os.path.join(args.output_dir, "best_model.pt"))
            print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch+1} epochs without improvement.")
            break
    
    # Save final model
    model.save_weights(os.path.join(args.output_dir, "final_model.pt"))
    
    # Load the best model for final evaluation
    model.load_weights(os.path.join(args.output_dir, "best_model.pt"))
    
    # Evaluate on test set
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, args.output_dir)
    
    # Visualize some predictions
    visualize_predictions(model, test_images, test_labels, device, args.output_dir)
    
    print(f"Training completed. Best validation accuracy: {best_val_accuracy:.2f}%, Test accuracy: {test_accuracy:.2f}%")
    
    return test_accuracy

def evaluate(model, data_loader, criterion, device):
    """Evaluate the model on the provided data loader."""
    model.eval()
    loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            batch_loss = criterion(outputs, targets)
            
            # Statistics
            loss += batch_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    # Calculate metrics
    avg_loss = loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def plot_training_curves(train_losses, val_losses, train_accuracies, val_accuracies, output_dir):
    """Plot training and validation curves."""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train')
    plt.plot(val_accuracies, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
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
    
    # Calculate confidence
    confidence = torch.max(predictions, dim=1)[0].cpu()
    
    # Plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i in range(num_samples):
        # Plot image
        image = sample_images[i, 0].cpu().numpy()
        axes[i].imshow(image, cmap='gray')
        
        # Set title (green if correct, red if wrong)
        title_color = 'green' if predicted_labels[i] == sample_labels[i] else 'red'
        axes[i].set_title(f"True: {sample_labels[i].item()}\nPred: {predicted_labels[i].item()}\nConf: {confidence[i]:.2f}", 
                         color=title_color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'predictions.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Train improved CNN for MNIST digit classification')
    
    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='data', 
                        help='directory containing the MNIST data')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=30, 
                        help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='weight decay (L2 penalty)')
    parser.add_argument('--dropout_rate', type=float, default=0.3, 
                        help='dropout rate for fully connected layers')
    parser.add_argument('--patience', type=int, default=10, 
                        help='number of epochs without improvement before early stopping')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='number of worker threads for data loading')
    parser.add_argument('--seed', type=int, default=42, 
                        help='random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output_improved', 
                        help='directory to save results')
    
    args = parser.parse_args()
    print(args)
    
    # Start training
    train(args)

if __name__ == "__main__":
    main() 