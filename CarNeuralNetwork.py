import numpy as np


class CarNeuralNetwork:
    def __init__(self, input_size=5, hidden_size=8, output_size=2):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    def forward(self, inputs):
        # inputs shape: (1, 5)
        z1 = np.dot(inputs, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = np.dot(a1, self.W2) + self.b2
        outputs = np.tanh(z2)   # Scale between -1 and 1
        return outputs[0]

    def mutate(self, rate=0.1, magnitude=0.2):
        child = CarNeuralNetwork()
        child.W1 = self.W1.copy()
        child.b1 = self.b1.copy()
        child.W2 = self.W2.copy()
        child.b2 = self.b2.copy()
        
        for w in [child.W1, child.b1, child.W2, child.b2]:
            mask = np.random.rand(*w.shape) < rate
            w += mask * np.random.randn(*w.shape) * magnitude
        return child