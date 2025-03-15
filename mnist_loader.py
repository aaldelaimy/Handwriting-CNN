import os
import gzip
import numpy as np
import struct
import shutil
from tqdm import tqdm

def download_mnist(data_dir='data'):
    """
    Download the MNIST dataset using a more direct approach.
    """
    import urllib.request
    
    # Create data directory
    os.makedirs(data_dir, exist_ok=True)
    
    # Define URLs for MNIST dataset from a very reliable source
    urls = {
        'train-images-idx3-ubyte.gz': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz': 'https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz': 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz': 'https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz'
    }
    
    # Download files
    for filename, url in urls.items():
        filepath = os.path.join(data_dir, filename)
        
        # Remove any corrupted files
        if os.path.exists(filepath):
            try:
                with gzip.open(filepath, 'rb') as f:
                    f.read(4)  # Just check if it's a valid gzip file
            except Exception:
                print(f"Removing corrupted file: {filepath}")
                os.remove(filepath)
        
        # Download the file if it doesn't exist
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            try:
                # Simple direct download with progress bar
                with urllib.request.urlopen(url) as response:
                    total_size = int(response.info().get('Content-Length', 0))
                    block_size = 8192
                    
                    with open(filepath, 'wb') as out_file:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                            while True:
                                buffer = response.read(block_size)
                                if not buffer:
                                    break
                                out_file.write(buffer)
                                pbar.update(len(buffer))
                
                # Verify the downloaded file
                with gzip.open(filepath, 'rb') as f:
                    f.read(4)  # Just check if it's a valid gzip file
                print(f"Downloaded {filename} successfully")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                if os.path.exists(filepath):
                    os.remove(filepath)
                return False
    
    print("All MNIST files downloaded successfully")
    return True

def load_mnist(data_dir='data', normalize=True, flatten=False):
    """Load the MNIST dataset."""
    # Make sure the data exists
    if not download_mnist(data_dir):
        raise RuntimeError("Failed to download MNIST dataset")
    
    # Define file paths
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    
    # Load images function
    def _load_images(path):
        with gzip.open(path, 'rb') as f:
            # Read file header
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            # Read image data
            buf = f.read()
            images = np.frombuffer(buf, dtype=np.uint8)
            return images.reshape(num, rows, cols)
    
    # Load labels function
    def _load_labels(path):
        with gzip.open(path, 'rb') as f:
            # Read file header
            magic, num = struct.unpack(">II", f.read(8))
            # Read label data
            buf = f.read()
            return np.frombuffer(buf, dtype=np.uint8)
    
    try:
        # Load images and labels
        train_images = _load_images(train_images_path)
        test_images = _load_images(test_images_path)
        train_labels = _load_labels(train_labels_path)
        test_labels = _load_labels(test_labels_path)
        
        # Normalize pixel values to [0, 1]
        if normalize:
            train_images = train_images / 255.0
            test_images = test_images / 255.0
        
        # Reshape images for CNN
        if flatten:
            train_images = train_images.reshape(-1, 28*28)
            test_images = test_images.reshape(-1, 28*28)
        else:
            # Add channel dimension for CNNs (N, 28, 28) -> (N, 1, 28, 28)
            train_images = train_images.reshape(-1, 1, 28, 28)
            test_images = test_images.reshape(-1, 1, 28, 28)
        
        # Convert labels to one-hot encoding
        def to_one_hot(labels, num_classes=10):
            one_hot = np.zeros((labels.shape[0], num_classes))
            one_hot[np.arange(labels.shape[0]), labels] = 1
            return one_hot
        
        train_labels_one_hot = to_one_hot(train_labels)
        test_labels_one_hot = to_one_hot(test_labels)
        
        return (train_images, train_labels, train_labels_one_hot), (test_images, test_labels, test_labels_one_hot)
    
    except Exception as e:
        print(f"Error loading MNIST data: {e}")
        # If loading fails, delete the data directory to force a fresh download next time
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
            print(f"Removed data directory due to loading error. Please run again.")
        raise RuntimeError(f"Failed to load MNIST dataset: {e}")

def get_batch(images, labels, batch_size, shuffle=True):
    """Generate batches from the dataset."""
    n_samples = images.shape[0]
    
    if shuffle:
        # Create a random permutation of indices
        indices = np.random.permutation(n_samples)
        images = images[indices]
        labels = labels[indices]
    
    # Yield batches
    for i in range(0, n_samples, batch_size):
        batch_x = images[i:i + batch_size]
        batch_y = labels[i:i + batch_size]
        yield batch_x, batch_y

if __name__ == "__main__":
    # Test the loader
    try:
        print("Loading MNIST dataset...")
        (train_images, train_labels, train_labels_one_hot), (test_images, test_labels, test_labels_one_hot) = load_mnist()
        
        print(f"Training set shape: {train_images.shape}")
        print(f"Test set shape: {test_images.shape}")
        print(f"Training labels shape: {train_labels.shape}")
        print(f"Training one-hot labels shape: {train_labels_one_hot.shape}")
        
        # Test batch generator
        batch_size = 32
        batches = list(get_batch(train_images, train_labels_one_hot, batch_size))
        print(f"Number of batches: {len(batches)}")
        print(f"Batch shapes: {batches[0][0].shape}, {batches[0][1].shape}")
        
        # Print sample image
        print("\nSample MNIST digit:")
        sample_idx = 42  # Just pick a sample
        sample_digit = train_labels[sample_idx]
        sample_image = train_images[sample_idx, 0]  # Get the first channel
        
        # Print ASCII representation of the digit
        for i in range(28):
            line = ""
            for j in range(28):
                if sample_image[i, j] > 0.5:
                    line += "██"
                elif sample_image[i, j] > 0.25:
                    line += "▓▓"
                elif sample_image[i, j] > 0:
                    line += "░░"
                else:
                    line += "  "
            print(line)
        print(f"Label: {sample_digit}")
        
        print("\nMNIST dataset loaded successfully!")
        
    except Exception as e:
        print(f"Error testing MNIST loader: {e}") 