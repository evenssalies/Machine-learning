## Perceptron
###   Evens Salies, v1: 02-12/2024

Perceptron for Binary Classification for the following data:

```math
x\equiv
    \left(
        \begin{array}{cc} 
            0 & 0 \\ 
            0 & 1 \\
            1 & 0 \\
            1 & 1
        \end{array}
    \right),\
w\equiv
    \left(
        \begin{array}{c}
        w_1 \\
        w_2
        \end{array}    
    \right),\
y\equiv
    \left(
        \begin{array}{c}
        0 \\
        0 \\
        0 \\
        1
        \end{array}    
    \right)
```

A bias, which is actually a third weight, $w_0$, is considered. It is a term added to the prediction.

```python
# perceptron.py
import numpy as np
np.random.seed(21041971)
```

The `__init__` constructor initializes the weights vector $w$ with random values and the constant to 0. The step size (`learning_rate` hyperparameter), has the usual value.

```python
class Perceptron:
    def __init__(self, input_size, learning_rate = 0.1):
        self.weights = np.random.rand(input_size)
        self.bias = 0
        self.learning_rate = learning_rate
```

The prediction threshold is set _ad hoc_ to 0, given the values of $x$, weights between 0 and 1, and the constant. For each example $i$, prediction $x'_iw+w_0$ is compared to the threshold. Training is done over an ajustable number of iterations. The optimization algorithm is similar to a __gradient descent__,

<p>
<div style="text-align: center;">
    `self.weights -= 2 * self.learning_rate * error * training_data[i]`.
</div>
<p>
 
```python
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
```

Input data and class inheritance.

```python
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])
perceptron = Perceptron(input_size = 2)
```

Training.

```python
perceptron.train(training_data, labels, epochs = 1000)
```

Predictions calculations.

```python
for i in range(len(training_data)):
    prediction = perceptron.predict(training_data[i])
    weights_vector = perceptron.weights
    constant = perceptron.bias
    print(
        f"Observation: {training_data[i]}, prediction: {prediction}, coefficients: {weights_vector}, bias: {constant}"
        )
```

The Perceptron correctely predicts the labels in $y$. Errors are calculated as differences between labels and predictions; in vector notation, and using final estimated values: $\hat{w}'= (0.20, 0.43)$ and $w_0=-0.60$. For entry say $i=3$, the weithed sum is $\hat{y}_3=0.20\times 1+0.43\times 1-.60\simeq 0.03$, which is greater than the threshold value of 0. Therefore, `predict` returns 1; there is no error. At the optimum weights, the full prediction $\hat{y}$ and `error` vector $\hat{e}$ are:

```math
\hat{y}=
    \left(
        \begin{array}{r}
        -.06 \\
        -.17 \\
        -.40 \\
        .03
        \end{array}    
    \right),\
\hat{e}\equiv y-\hat{y}=
    \left(
        \begin{array}{l}
        0-0 \\
        0-0 \\
        0-0 \\
        1-1
        \end{array}    
    \right)=
        \left(
        \begin{array}{r}
        0 \\
        0 \\
        0 \\
        0
        \end{array}    
    \right).
```

```
Observation: [0 0], prediction: 0, coefficients: [0.20163156 0.42922185], bias: -0.6
Observation: [0 1], prediction: 0, coefficients: [0.20163156 0.42922185], bias: -0.6
Observation: [1 0], prediction: 0, coefficients: [0.20163156 0.42922185], bias: -0.6
Observation: [1 1], prediction: 1, coefficients: [0.20163156 0.42922185], bias: -0.6
```