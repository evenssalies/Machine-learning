# Perceptron
#   Evens Salies, v1: 02-12/2024

import numpy as np
np.random.seed(21041971)

class Perceptron:
    def __init__(self, input_size, learning_rate = 0.1):
        self.weights = np.random.rand(input_size)
        self.bias = 0
        self.learning_rate = learning_rate

    def predict(self, inputs, threshold = 0):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if weighted_sum > threshold else 0

    def train(self, training_data, labels, epochs):
        for _ in range(epochs):
            for i in range(len(training_data)):
                prediction = self.predict(training_data[i])
                error = labels[i] - prediction
                self.weights += self.learning_rate * error * training_data[i]
                self.bias += self.learning_rate * error

training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])

perceptron = Perceptron(input_size = 2)
perceptron.train(training_data, labels, epochs = 1000)

print("\n")
for i in range(len(training_data)):
    prediction = perceptron.predict(training_data[i])
    weights_vector = perceptron.weights
    constant = perceptron.bias
    print(
        f"Observation: {training_data[i]}, prediction: {prediction}, coefficients: {weights_vector}, bias: {constant}"
        )
