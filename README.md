# MNIST Digit Classification with CNN 

This project implements two approaches for handwritten digit classification using Convolutional Neural Networks (CNNs) on the MNIST dataset:
1. A standard PyTorch CNN implementation
2. An improved CNN architecture with advanced training techniques

## Project Features

- **Two CNN implementations**:
  - Basic PyTorch CNN with ~95% accuracy
  - Improved architecture with ~98% accuracy
- **Advanced techniques** including batch normalization, dropout, and data augmentation
- **Complete training pipeline** with learning rate scheduling and early stopping
- **Comprehensive visualization suite** including confusion matrices and error analysis
- **Support for custom images** to test the trained models


## Model Architectures

### Basic CNN Architecture
```
CNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1))
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2)
  (fc1): Linear(in_features=1600, out_features=128)
  (fc2): Linear(in_features=128, out_features=10)
)
```

### Improved CNN Architecture
```
ImprovedCNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn1): BatchNorm2d(32)
  (pool1): MaxPool2d(kernel_size=2, stride=2)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn2): BatchNorm2d(64)
  (pool2): MaxPool2d(kernel_size=2, stride=2)
  (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (bn3): BatchNorm2d(128)
  (pool3): MaxPool2d(kernel_size=2, stride=2)
  (fc1): Linear(in_features=1152, out_features=256)
  (bn4): BatchNorm1d(256)
  (dropout1): Dropout(p=0.25)
  (fc2): Linear(in_features=256, out_features=10)
)
```

## Setup and Installation

**Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Models

#### Basic CNN:
```bash
python train_pytorch.py --epochs 15 --batch_size 64 --learning_rate 0.001 --output_dir output_pytorch
```

#### Improved CNN:
```bash
python train_pytorch_improved.py --epochs 15 --batch_size 128 --output_dir output_improved
```

Parameters:
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Initial learning rate
- `--output_dir`: Directory to save results
- `--weight_decay`: L2 regularization strength (improved model)
- `--dropout_rate`: Dropout probability (improved model)

### Testing the Models

#### Basic CNN:
```bash
python test_pytorch.py --model_path output_pytorch/best_model.pt --visualize mnist
```

#### Improved CNN:
```bash
python test_pytorch_improved.py --model_path output_improved/best_model.pt --visualize mnist
```

For custom images:
```bash
python test_pytorch_improved.py --model_path output_improved/best_model.pt image --image_path path/to/your/digit.png --visualize
```

## Enhanced Features in the Improved Model

### 1. Architectural Improvements
- **Deeper Network**: Three convolutional blocks instead of two
- **Batch Normalization**: After each convolutional layer to stabilize training
- **Dropout Regularization**: To prevent overfitting
- **Kaiming Initialization**: Better weight initialization for faster convergence

### 2. Training Optimizations
- **Data Augmentation**: Random affine transformations and erasing
- **Learning Rate Scheduling**: Adaptive learning rate based on validation performance
- **Early Stopping**: Prevents overfitting by monitoring validation metrics
- **Gradient Clipping**: Prevents gradient explosion
- **Train/Validation Split**: Better monitoring of generalization performance

## Results

### Performance Comparison
| Model | Test Accuracy | Parameters | Training Time |
|-------|---------------|------------|--------------|
| Basic CNN | ~95% | 122,570 | 15 minutes |
| Improved CNN | ~98% | 391,370 | 25 minutes |

### Visualizations

The improved model generates several visualizations to analyze performance:

#### Training Curves
![Training Curves](output_improved/training_curves.png)

#### Confusion Matrix
![Confusion Matrix](output_improved/test_results/confusion_matrix.png)

#### Sample Predictions
![Predictions](output_improved/test_results/predictions.png)

#### Misclassified Examples
![Misclassified Examples](output_improved/test_results/misclassified_examples.png)

## Implementation Details

### Key Components of the Improved Model

#### Batch Normalization
Normalizes the inputs of each layer to reduce internal covariate shift, allowing:
- Higher learning rates
- Less careful initialization
- Regularization effect

#### Data Augmentation
Implemented through the `AugmentedMNIST` class which applies:
- Random rotations (±5°)
- Random translations (±10%)
- Random scaling (90-110%)
- Random erasing (simulated occlusion)

#### Learning Rate Scheduling
Uses `ReduceLROnPlateau` to adjust learning rate when validation metrics plateau:
```python
scheduler = ReduceLROnPlateau(
    optimizer, mode='max', factor=0.1, patience=3, verbose=True
)
 