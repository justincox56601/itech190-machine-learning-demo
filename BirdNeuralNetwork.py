import numpy as np

class BirdNeuralNetwork:
    def __init__(self, input_size=3, hidden_size=6, output_size=1):
        # Small weight initialization so initial actions aren't maxed out
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))

    def forward(self, inputs):
        # inputs shape: (1, 3) -> [vel, dist_x, dist_y]
        z1 = np.dot(inputs, self.W1) + self.b1
        a1 = np.maximum(0, z1)  # ReLU
        z2 = np.dot(a1, self.W2) + self.b2
        # Sigmoid activation for binary jump decision (0.0 to 1.0)
        output = 1.0 / (1.0 + np.exp(-z2))
        return output[0][0]

    def mutate(self, rate=0.15, magnitude=0.3):
        child = BirdNeuralNetwork()
        child.W1 = self.W1.copy()
        child.b1 = self.b1.copy()
        child.W2 = self.W2.copy()
        child.b2 = self.b2.copy()

        for w in [child.W1, child.b1, child.W2, child.b2]:
            mask = np.random.rand(*w.shape) < rate
            w += mask * np.random.randn(*w.shape) * magnitude
        return child
