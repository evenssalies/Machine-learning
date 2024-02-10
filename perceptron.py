# Chat GPT me propose le perceptron suivant.

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if weighted_sum > 0 else 0

    def train(self, training_data, labels, epochs):
        for _ in range(epochs):
            for i in range(len(training_data)):
                prediction = self.predict(training_data[i])
                error = labels[i] - prediction
                self.weights += self.learning_rate * error * training_data[i]
                self.bias += self.learning_rate * error

# Example usage:
if __name__ == "__main__":
    training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=2)
    perceptron.train(training_data, labels, epochs=1000)

    for i in range(len(training_data)):
        prediction = perceptron.predict(training_data[i])
        print(f"Input: {training_data[i]}, Predicted: {prediction}")
