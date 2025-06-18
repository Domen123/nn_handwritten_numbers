# Handwritten Digit Recognition Neural Network

A simple neural network implementation for recognizing handwritten digits (0-9) using the MNIST dataset.

## Features

- **Neural Network**: 2-layer feedforward network (784 → 256 → 10)
- **Training**: Mini-batch gradient descent with cross-entropy loss
- **Web Interface**: Interactive drawing canvas for real-time predictions
- **Model Persistence**: Save and load trained weights

## Architecture

- **Input Layer**: 784 neurons (28×28 pixel images)
- **Hidden Layer**: 256 neurons with sigmoid activation
- **Output Layer**: 10 neurons with softmax activation (digits 0-9)

## Setup and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Run the training script to train the neural network and save the weights:

```bash
python train_and_save.py
```

This will:

- Load the MNIST dataset
- Train the neural network for 1000 epochs
- Save the trained weights to `model/` directory
- Verify the saved weights work correctly

### 3. Use the Web Interface

After training, open `index.html` in your web browser:

```bash
# On macOS/Linux
open index.html

# Or simply double-click the file in your file explorer
```

### 4. Draw and Predict

1. Draw a digit (0-9) on the canvas
2. Click "Predict" to see the neural network's prediction
3. Click "Clear" to start over

## Files Structure

```
handwritten-numbers-nn/
├── neural_network.py      # Core neural network implementation
├── train_and_save.py      # Training script that saves weights
├── index.html             # Web interface
├── requirements.txt       # Python dependencies
				numpy>=1.21.0
				tensorflow>=2.8.0
				matplotlib>=3.5.0
				scikit-learn>=1.1.0
				pandas>=1.4.0 

├── model/                 # Trained weights (created after training)
│   ├── W1.json           # Hidden layer weights
│   ├── b1.json           # Hidden layer biases
│   ├── W2.json           # Output layer weights
│   └── b2.json           # Output layer biases
└── README.md             # This file
```

## Technical Details

### Activation Functions

- **Hidden Layer**: Sigmoid function for non-linearity
- **Output Layer**: Softmax function for probability distribution

### Loss Function

- **Cross-entropy loss** for multi-class classification

### Training

- **Optimizer**: Mini-batch gradient descent
- **Batch Size**: 32
- **Learning Rate**: 0.1
- **Epochs**: 1000

### Data Preprocessing

- Images resized to 28×28 pixels
- Pixel values normalized to [0, 1]
- Labels converted to one-hot encoding

## Troubleshooting

### "Model files not found" Error

If you see this error in the web interface:

1. Make sure you've run `python train_and_save.py` first
2. Check that the `model/` directory exists with the weight files
3. Ensure you're serving the files from a web server (not just opening the HTML file)

### Training Issues

- If training is too slow, reduce the number of epochs in `train_and_save.py`
- If accuracy is poor, try adjusting the learning rate or hidden layer size

## Performance

The model typically achieves:

- Training accuracy: ~95%+ on MNIST training set
- Reasonable performance on hand-drawn digits

## Future Improvements

- Add data augmentation for better generalization
- Implement dropout for regularization
- Add convolutional layers for better feature extraction
- Support for different model architectures
- Real-time training visualization
