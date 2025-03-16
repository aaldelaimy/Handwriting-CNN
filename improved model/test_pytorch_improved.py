import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model_pytorch_improved import ImprovedCNN
from train_pytorch_improved import load_mnist_pytorch, evaluate

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Test CNN on MNIST or custom images')
    
    # Common parameters
    parser.add_argument('--model_path', type=str, default='output_improved/best_model.pt', 
                        help='path to the model weights')
    parser.add_argument('--output_dir', type=str, default='output_improved/test_results', 
                        help='directory to save results')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for testing')
    parser.add_argument('--visualize', action='store_true', 
                        help='visualize predictions')
    parser.add_argument('--force_cpu', action='store_true', 
                        help='force using CPU even if GPU is available')
    
    # Create subparsers for different test modes
    subparsers = parser.add_subparsers(dest='mode', help='test mode')
    
    # MNIST test set parser
    mnist_parser = subparsers.add_parser('mnist', help='test on MNIST test set')
    mnist_parser.add_argument('--num_samples', type=int, default=10, 
                              help='number of samples to visualize')
    
    # Custom image parser
    image_parser = subparsers.add_parser('image', help='test on a custom image')
    image_parser.add_argument('--image_path', type=str, required=True, 
                              help='path to the image file')
    image_parser.add_argument('--invert', action='store_true', 
                              help='invert image (for white digits on black background)')
    
    args = parser.parse_args()
    
    # Run appropriate test mode
    if args.mode == 'mnist':
        test_mnist(args)
    elif args.mode == 'image':
        test_custom_image(args)
    else:
        parser.print_help()

def test_mnist(args):
    """Test the trained model on MNIST test set."""
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load MNIST test set
    print("Loading MNIST test set...")
    _, (test_images, test_labels) = load_mnist_pytorch()
    print(f"Test images shape: {test_images.shape}")
    
    # Create and load model
    model = ImprovedCNN(dropout_rate=0.0).to(device)  # No dropout needed for inference
    
    try:
        model.load_weights(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test on MNIST
    model.eval()
    correct = 0
    total = 0
    
    # Set up a data loader for batch processing
    test_dataset = torch.utils.data.TensorDataset(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False
    )
    
    # Initialize confusion matrix
    confusion_matrix = torch.zeros(10, 10, dtype=torch.long)
    
    # Process in batches
    print("Testing model...")
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update confusion matrix
            for t, p in zip(targets.view(-1), predicted.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
    
    # Calculate accuracy
    test_accuracy = 100.0 * correct / total
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Save confusion matrix
    plot_confusion_matrix(confusion_matrix, args.output_dir)
    
    # Visualize predictions
    if args.visualize:
        visualize_predictions(model, test_images, test_labels, device, args.output_dir, args.num_samples)
        visualize_errors(model, test_images, test_labels, device, args.output_dir)

def plot_confusion_matrix(confusion_matrix, output_dir):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix.numpy(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = [str(i) for i in range(10)]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = confusion_matrix.max() / 2.
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, format(confusion_matrix[i, j].item(), 'd'),
                     horizontalalignment="center",
                     color="white" if confusion_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Confusion matrix saved to {os.path.join(output_dir, 'confusion_matrix.png')}")

def visualize_errors(model, images, labels, device, output_dir, num_samples=10):
    """Visualize misclassified examples."""
    model.eval()
    
    # Get predictions for all test images
    all_images = images.to(device)
    batch_size = 1000  # Process in batches to avoid memory issues
    predictions = []
    
    with torch.no_grad():
        for i in range(0, len(all_images), batch_size):
            batch = all_images[i:i+batch_size]
            batch_preds = model.predict(batch)
            predictions.append(batch_preds)
    
    # Combine predictions
    all_predictions = torch.cat(predictions, dim=0).cpu()
    predicted_labels = torch.argmax(all_predictions, dim=1)
    
    # Find misclassified examples
    incorrect_indices = (predicted_labels != labels).nonzero(as_tuple=True)[0]
    
    if len(incorrect_indices) == 0:
        print("No misclassified examples found!")
        return
    
    # Select a random subset of misclassified examples
    if len(incorrect_indices) > num_samples:
        indices = incorrect_indices[torch.randperm(len(incorrect_indices))[:num_samples]]
    else:
        indices = incorrect_indices
    
    # Plot the misclassified examples
    n_cols = 5
    n_rows = (len(indices) + n_cols - 1) // n_cols  # Ceiling division
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        # Get image and predictions
        image = images[idx, 0].cpu().numpy()
        true_label = labels[idx].item()
        pred_label = predicted_labels[idx].item()
        confidence = all_predictions[idx, pred_label].item()
        
        # Plot image
        ax.imshow(image, cmap='gray')
        ax.set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}", color='red')
        ax.axis('off')
    
    # Hide any unused subplots
    for i in range(len(indices), n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'misclassified_examples.png'))
    plt.close()
    print(f"Misclassified examples saved to {os.path.join(output_dir, 'misclassified_examples.png')}")

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
    print(f"Predictions visualization saved to {os.path.join(output_dir, 'predictions.png')}")

def test_custom_image(args):
    """Test the trained model on a custom image."""
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and load model
    model = ImprovedCNN(dropout_rate=0.0).to(device)  # No dropout needed for inference
    
    try:
        model.load_weights(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load and preprocess image
    try:
        # Open and convert to grayscale
        image = Image.open(args.image_path).convert('L')
        
        # Define preprocessing
        transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # Apply preprocessing
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Invert if needed (for white digits on black background)
        if args.invert:
            image_tensor = 1.0 - image_tensor
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Make prediction
    with torch.no_grad():
        model.eval()
        probs = model.predict(image_tensor)
        
    # Get predictions
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_class].item()
    
    # Print results
    print(f"Predicted digit: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
    # Show all class probabilities
    print("\nClass probabilities:")
    for i in range(10):
        print(f"  Digit {i}: {probs[0, i].item():.4f}")
    
    # Visualize prediction
    if args.visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.title(f"Predicted: {predicted_class} (Confidence: {confidence:.4f})")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'custom_prediction.png'))
        plt.close()
        
        print(f"Prediction visualization saved to {os.path.join(args.output_dir, 'custom_prediction.png')}") 