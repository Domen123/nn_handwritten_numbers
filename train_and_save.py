#!/usr/bin/env python3
"""
Script to train the neural network and save the weights for the web interface.
Run this script to train the model and save the weights to the model/ directory.
"""

from neural_network import NeuralNetwork, load_mnist_data
import numpy as np
import os

def main():
    print("Training Neural Network for Handwritten Digit Recognition...")
    
    # Initialize network
    nn = NeuralNetwork()
    print("Neural network initialized with architecture: 784 -> 256 -> 10")

    # Load and preprocess data
    print("Loading MNIST dataset...")
    X, y = load_mnist_data()
    print(f"Training data shape: X={X.shape}, y={y.shape}")

    # Train the network
    print("Starting training...")
    nn.train(X, y, epochs=1000, learning_rate=0.1, batch_size=32)

    # Save the trained weights
    print("Saving trained weights...")
    nn.save_weights()

    # Test prediction to verify the model works
    print("Testing the trained model...")
    test_image = X[:, 0:1]  # Example: first image
    prediction = nn.forward(test_image)
    predicted_digit = np.argmax(prediction)
    print(f"Test prediction: {predicted_digit}")
    
    # Test loading weights to ensure they were saved correctly
    print("Verifying saved weights...")
    nn2 = NeuralNetwork()
    nn2.load_weights()
    prediction2 = nn2.forward(test_image)
    predicted_digit2 = np.argmax(prediction2)
    print(f"Loaded model prediction: {predicted_digit2}")
    
    if predicted_digit == predicted_digit2:
        print("✅ Weights saved and loaded successfully!")
    else:
        print("❌ Error: Predictions don't match after loading weights")
    
    print("\n🎉 Training complete! You can now use the web interface.")
    print("Open index.html in your browser to test the model.")

if __name__ == "__main__":
    main() 