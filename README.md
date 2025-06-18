# Handwritten Digit Recognition Neural Network

A simple neural network implementation for recognizing handwritten digits using the MNIST dataset. This project demonstrates a complete neural network from scratch using NumPy, including forward propagation, backpropagation, and mini-batch gradient descent.

## Features

- **From-scratch implementation**: No deep learning frameworks used for the core neural network
- **MNIST dataset**: Trained on the standard handwritten digit recognition dataset
- **Interactive web interface**: Draw digits and get predictions in real-time
- **Mini-batch gradient descent**: Efficient training with configurable batch sizes
- **Cross-entropy loss**: Proper loss function for multi-class classification

## Architecture

The neural network has the following architecture:
- **Input layer**: 784 neurons (28×28 flattened images)
- **Hidden layer**: 256 neurons with sigmoid activation
- **Output layer**: 10 neurons with softmax activation (digits 0-9)

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Create and activate conda environment
conda create -n text_nn python=3.9 -y
conda activate text_nn

# Install dependencies
conda install numpy tensorflow matplotlib scikit-learn pandas -y
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv text_nn
source text_nn/bin/activate  # On Windows: text_nn\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training the Neural Network

```bash
python neural_network.py
```

This will:
1. Load the MNIST dataset
2. Train the neural network for 1000 epochs
3. Display training progress every 100 epochs
4. Make a prediction on the first training image

### Web Interface

Open `index.html` in a web browser to:
- Draw digits on a canvas
- Get real-time predictions
- View prediction probabilities

## Project Structure

handwritten-numbers-nn/
├── neural_network.py # Main neural network implementation
├── index.html # Web interface for drawing digits
├── requirements.txt # Python dependencies
└── README.md # This file



## Implementation Details

### Activation Functions
- **Sigmoid**: Used in hidden layer for non-linearity
- **Softmax**: Used in output layer for multi-class probabilities

### Loss Function
- **Cross-entropy loss**: Proper loss function for classification tasks

### Training
- **Mini-batch gradient descent**: Configurable batch size (default: 32)
- **Learning rate**: Configurable (default: 0.1)
- **Epochs**: Configurable (default: 1000)

## Performance

The neural network typically achieves:
- Training accuracy: ~95%+ after 1000 epochs
- Test accuracy: ~94%+ on MNIST test set

## Customization

You can modify the network architecture by changing:
- `hidden_size` in the `NeuralNetwork` constructor
- `learning_rate` and `batch_size` in the training call
- Number of `epochs` for training

## Dependencies

- **NumPy**: Core numerical computations
- **TensorFlow**: MNIST dataset loading
- **Matplotlib**: Optional for visualization
- **scikit-learn**: Optional for evaluation metrics
- **pandas**: Optional for data manipulation

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues and enhancement requests!