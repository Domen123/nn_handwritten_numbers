import numpy as np

# Sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid for backpropagation
def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

# Softmax for output layer
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x, axis=0)

# Cross-entropy loss
def cross_entropy_loss(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-15))  # Add small epsilon to avoid log(0)

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        # Initialize weights and biases with small random values
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def forward(self, X):
        # Forward propagation
        # Input to hidden: z1 = W1 * X + b1, a1 = sigmoid(z1)
        self.z1 = np.dot(self.W1, X) + self.b1
        self.a1 = sigmoid(self.z1)
        # Hidden to output: z2 = W2 * a1 + b2, a2 = softmax(z2)
        self.z2 = np.dot(self.W2, self.a1) + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate=0.01):
        # Backpropagation
        m = X.shape[1]  # Number of examples

        # Gradient of loss w.r.t. output (cross-entropy with softmax)
        dz2 = output - y  # y is one-hot encoded
        dW2 = np.dot(dz2, self.a1.T) / m
        db2 = np.sum(dz2, axis=1, keepdims=True) / m

        # Gradient for hidden layer
        da1 = np.dot(self.W2.T, dz2)
        dz1 = da1 * sigmoid_derivative(self.z1)
        dW1 = np.dot(dz1, X.T) / m
        db1 = np.sum(dz1, axis=1, keepdims=True) / m

        # Update weights and biases
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs=1000, learning_rate=0.01, batch_size=32):
        m = X.shape[1]  # Number of examples

        for epoch in range(epochs):
            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                X_batch = X[:, i:i+batch_size]
                y_batch = y[:, i:i+batch_size]

                # Forward pass
                output = self.forward(X_batch)

                # Backward pass
                self.backward(X_batch, y_batch, output, learning_rate)

            # Compute loss for monitoring
            if epoch % 100 == 0:
                output = self.forward(X)
                loss = cross_entropy_loss(output, y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

def load_mnist_data():
    from tensorflow.keras.datasets import mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).T / 255.0
    y_train = np.eye(10)[y_train].T
    return X_train, y_train

if __name__ == "__main__":
    # Initialize network
    nn = NeuralNetwork()

    # Load and preprocess data
    X, y = load_mnist_data()
    print(f"Training data shape: X={X.shape}, y={y.shape}")

    # Train the network
    nn.train(X, y, epochs=1000, learning_rate=0.1, batch_size=32)

    # Test prediction
    test_image = X[:, 0:1]  # Example: first image
    prediction = nn.forward(test_image)
    predicted_digit = np.argmax(prediction)
    print(f"Predicted digit: {predicted_digit}")