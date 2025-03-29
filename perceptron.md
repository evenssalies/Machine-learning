## Perceptron

### Frank Rosenblatt

`Perceptron' is a term coined by Frank Rosenblatt in the 1950s. In the study of pattern recognition, the **Perceptron** originally was a computer program that reproduces a nerve net system consisting of *data* (e.g., the information received by a retina), a *model* (an association area), and a *classifier* (one or more response units making predictions).[^1] Such program was quickly used for **artificial intelligence** in different researches, whereas Rosenblatt aimed at *"investigating the physical structures and neurodynamic principles which underlie 'natural intelligence'*" [...]; it is a "*brain model*" wrote Rosenblatt, "*not an invention for pattern recognition.*" Rosenblatt wanted to be useful to physiological psychologist.[^2]

[^1]: White, B. W. (1963). Review of `Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms', by F. Rosenblatt. **The American Journal of Psychology**, 76(4), 705–707.
[^2]: These quotes are taken from a 1961 book by Rosenblatt [Perceptrons and the Theory of Brain Mechanisms](ishttps://safari.ethz.ch/digitaltechnik/spring2018/lib/exe/fetch.php?media=neurodynamics1962rosenblatt.pdf). I did not read more than that part of the book, neither the first paper where one can find "Perceptron" would be [Rosenblatt, F. (1957). The perceptron &#8209; A perceiving and recognizing automaton. Cornell Aeronautical Laboratory Report No. 85-460-1](https://bpb-us-e2.wpmucdn.com/websites.umass.edu/dist/a/27637/files/2016/03/rosenblatt-1957.pdf). I wish I could find the time to do this some day ... when I'll retire 👴.

There is an example of a Perceptron for binary classification on the following data:

```math
x\equiv
    \left(
        \begin{array}{cc} 0 & 0 \\ 0 & 1 \\ 1 & 0 \\ 1 & 1 \end{array}
    \right),\
w\equiv
    \left(
        \begin{array}{c} w_1 \\ w_2 \end{array}    
    \right),\
y\equiv
    \left(
        \begin{array}{c} 0 \\ 0 \\ 0 \\ 1 \end{array}    
    \right)
```

A bias, which is actually a third weight, $w_0$, is considered. It is a term added to the prediction. Import the library and set the seed for pseudo-random number generation (this is useful for identical replications of the numbers created in this example):

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

We are going to identify the model by updating the weigths a thousand times. We could try inputing different *ad hoc* values instead of using an updating rule. Prediction enters a Heaviside step function.[^3] This latter is plausible given the values of $y$ and $x$. Optimum weights will eventually be between 0 and 1. For each example $i$, prediction $x'_iw+w_0$ is compared to the threshold. If the model prediction is positive it is set to 1, but 0 otherwise.

Training is done over an ajustable number of iterations. The optimization algorithm (updating rule) below is different from a __gradient descent__. It is a simple update rule based on error. The rule for a gradient descent would be ```self.weights -= 2 * self.learning_rate * error * training_data[i]```, with weights (and constant) that are updated by moving in the direction that reduces the loss function (I'll write the maths later).

Assume positive weigths and ignore the learning rate for the moment. In the Perceptron, the error $\hat{e}_i$ is $y_i-H(\hat{y}_i)$. If the error is negative, then this is because the weights are to high; therefore, we should decrease the weigths. The rule is ```self.weights += self.learning_rate * error * training_data[i]```, and the constant is updated accordingly.

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

The Perceptron correctely predicts the labels in $y$. Errors are calculated as differences between labels and predictions. Final estimated values are $\hat{w}=(0.20, 0.43)'$ and $w_0=-0.60$. For entry say $i=3$, the weithed sum is $\hat{y}_3=0.20\times 1+0.43\times 1-.60\simeq 0.03$, which is greater than the threshold value of 0. Therefore, `predict` returns 1; there is no error. At the optimum weights, the full prediction $\hat{y}$ and `error` vector $\hat{e}\equiv y-\hat{y}$ are:

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
\hat{e}=
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
[^3]: The Heaviside step function would take the value 1 for prediction greater than or equal to 0. In our Perceptron, the inequality is strict.

###   Evens Salies, v1: 02-12/2024
