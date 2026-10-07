import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim=2, hidden_dim=4, output_dim=3, learning_rate=0.05):
        self.learning_rate = learning_rate
        #np.random.seed(42) #uncomment this to have consistent starting values
        
        # Xavier/He Initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1, output_dim))
        
        self.history = []

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

    def train(self, X, y_onehot, y_true, epochs=80):
        # Save Initial State (Epoch 0)
        preds = self.predict(X)
        errors = np.sum(preds != y_true)
        self.history.append((self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(), 0, errors))

        for epoch in range(1, epochs + 1):
            # 1. Forward Pass
            output = self.forward(X)
            
            # 2. Backpropagation
            # Gradient on Output Layer (Cross-Entropy Loss derivative with Softmax)
            dZ2 = (output - y_onehot) / len(X)
            dW2 = np.dot(self.A1.T, dZ2)
            db2 = np.sum(dZ2, axis=0, keepdims=True)
            
            # Gradient on Hidden Layer
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.relu_derivative(self.Z1)
            dW1 = np.dot(X.T, dZ1)
            db1 = np.sum(dZ1, axis=0, keepdims=True)
            
            # 3. Update Weights & Biases
            self.W1 -= self.learning_rate * dW1
            self.b1 -= self.learning_rate * db1
            self.W2 -= self.learning_rate * dW2
            self.b2 -= self.learning_rate * db2
            
            # Record state every 2 epochs to keep animation smooth
            if epoch % 2 == 0 or epoch == epochs:
                preds = self.predict(X)
                errors = np.sum(preds != y_true)
                self.history.append((self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy(), epoch, errors))