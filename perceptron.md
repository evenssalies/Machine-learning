## Perceptron
###   Evens Salies, v1: 02-12/2024

Example of Perceptron for Binary Classification.

```python
# perceptron.py
import numpy as np
np.random.seed(21041971)
```

The `__init__` constructor initializes the weight vector $w$ with random values and sets the constant (bias) to 0. The step size (hyperparameter), `learning_rate`, has a default value.

```python
class Perceptron:
    def __init__(self, input_size, learning_rate = 0.1):
        self.weights = np.random.rand(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
```

The prediction threshold is set to 0 given the values of $x$, weights between 0 and 1, and the constant. For each example $i$, $w_i'x_i + \text{constant}$ is compared to the threshold. Training is done over a standard number of iterations (in each iteration, all training observations are processed). The optimization algorithm is similar to a __gradient descent__ `(self.weights -= 2 * self.learning_rate * error * training_data[i])`.

```python
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
```
Input data.

```python
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])
```

Class inheritance.

```python
perceptron = Perceptron(input_size = 2)
```

Take a look at the initial value of the constant.

```python
print("\n")
print(f"Initial constant: {perceptron.bias}\n")
```

Training, and final value of the constant.

```python
perceptron.train(training_data, labels, epochs = 1000)
print(f"Final constant: {perceptron.bias}\n")
```

Predictions.

```python
for i in range(len(training_data)):
    prediction = perceptron.predict(training_data[i])
    weights_vector = perceptron.weights
    print(f"Observation: {training_data[i]}, prediction: {prediction}, coefficient: {weights_vector}")
```