import urllib.request
import csv
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    
    def __init__(self, num_features, learning_rate=0.01, epochs=10, activation_fn='SIGN'):
        self.learning_rate = learning_rate
        self.epochs = epochs
        # Start with small random weights and zero bias
        #np.random.seed(42)  # Fixed seed so students get reproducible results in class
        self.weights = np.random.normal(loc=0.0, scale=0.01, size=num_features)
        self.bias = 1.0
        self.activation_fn = getattr(self, activation_fn.lower(), self.sign) #if I passs a non existent activation function, just use sign
        self.history = []

    def predict(self, input):
        return self.activation_fn(input)

    def train(self, features, labels):
        """
        Adjusts weights and bias whenever a prediction is wrong.
        """
        self.history.append((self.weights.copy(), self.bias, 0, "Start"))
        epoch = 0
        errors = 1
        while epoch < self.epochs and errors > 0:
            errors = 0
            epoch +=1 
            for i in range(len(features)):
                xi = features[i]
                target = labels[i]
                
                #prediction = self.predict(xi)
                prediction = self.activation_fn(xi)
                
                # Update rule: Only fires if target != prediction
                # If target == prediction, error_signal is 0 (no update)
                error_signal = target - prediction
                
                if error_signal != 0:
                    self.weights += self.learning_rate * error_signal * xi
                    self.bias += self.learning_rate * error_signal
                    errors += 1
            
            print(f"Epoch {epoch }/{self.epochs} - Misclassifications: {errors}")
            # Save state after each epoch pass
            self.history.append((self.weights.copy(), self.bias, epoch, errors))
            

    def get_history(self):
        return self.history
    
    def sign(self, features):
        """
        Calculates: dot(input, w) + b
        Returns +1 if >= 0, else -1
        """
        linear_combination = np.dot(features, self.weights) + self.bias
        return 1 if linear_combination >= 0 else -1