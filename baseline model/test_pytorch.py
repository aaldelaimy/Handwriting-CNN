import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from model_pytorch import CNN
from train_pytorch import load_mnist_pytorch

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
    model = CNN().to(device)
    
    try:
        model.load_weights(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Test on MNIST
    model.eval()
    correct = 0
    total = 0
    
    batch_size = 100
    num_batches = (test_images.size(0) - 1) // batch_size + 1
    
    # Calculate accuracy
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Testing"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, test_images.size(0))
            
            inputs = test_images[start_idx:end_idx].to(device)
            targets = test_labels[start_idx:end_idx].to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    test_accuracy = 100.0 * correct / total
    print(f"Test accuracy: {test_accuracy:.2f}%")
    
    # Visualize predictions
    if args.visualize:
        visualize_predictions(model, test_images, test_labels, device, args.output_dir, args.num_samples)

def test_custom_image(args):
    """Test the trained model on a custom image."""
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create and load model
    model = CNN().to(device)
    
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
        
    predicted_class = torch.argmax(probs, dim=1).item()
    confidence = probs[0, predicted_class].item()
    
    # Print results
    print(f"Predicted digit: {predicted_class}")
    print(f"Confidence: {confidence:.4f}")
    
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
    
    print(f"Predictions visualization saved to {os.path.join(output_dir, 'predictions.png')}")

def main():
    parser = argparse.ArgumentParser(description='Test CNN on MNIST or custom images')
    
    # Common parameters
    parser.add_argument('--model_path', type=str, default='output_pytorch/best_model.pt', 
                        help='path to the model weights')
    parser.add_argument('--output_dir', type=str, default='output_pytorch/test_results', 
                        help='directory to save results')
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

if __name__ == "__main__":
    main() 