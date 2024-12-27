# Perceptron
#   Evens Salies, v1: 02-12/2024

Exemple de Perceptron pour une classification dichotomique

```python
# perceptron.py
import numpy as np
np.random.seed(21041971)
```

Le constructeur `init` initialise le vecteur de poids $w$ et la constante (le biais) avec des valeurs aléatoires (la taille de $w$ est donnée par le tableau $x$ d'entrée). L'hyperparamètre de pas `learning_rate` est standard.

```python
class Perceptron:
    def __init__(self, input_size, learning_rate = 0.1):
        self.weights = np.random.rand(input_size)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
```

Le seuil de prédiction est fixé à 0 étant donné les valeurs de $x$, les poids entre 0 et 1 et la constante. Pour chaque exmple $i$, $w_i'x_i+constante$ est comparé à la valeur 0. L'entraînement se fait sur un nombre standard d'itérations (à chaque itération on parcourt toutes les observations d'entraînement). L'algorithme d'optimisation est proche d'une __descente de gradient__ `(self.weights -= 2*self.learning_rate * error * training_data[i])`.

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

Les données en entrée.

```python
training_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
labels = np.array([0, 0, 0, 1])
```

Héritage de classe.

```python
perceptron = Perceptron(input_size = 2)
```

Jette un oeil à la valeur de départ de la constante.

```python
print("\n")
print(f"Initial constant: {perceptron.bias}\n")
```

Entraînement, et valeur finale de la constante.

```python
perceptron.train(training_data, labels, epochs = 1000)
print(f"Final constant: {perceptron.bias}\n")
```

Prédicitons.

```python
for i in range(len(training_data)):
    prediction = perceptron.predict(training_data[i])
    weights_vector = perceptron.weights
    print(f"Observation: {training_data[i]}, prediction: {prediction}, coefficient: {weights_vector}")
```