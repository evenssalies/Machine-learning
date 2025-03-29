## Classification - Semi-Supervised learning (SSL)[^1]

### Evens Salies, v1: 01/2025

In ML, the concept of SSL is used in situations where $y_i$ is not observed unlike the $x_i$'s. In econometrics, such a situation occurs when $y_i$ is missing at random (MAR) or when there sample selection. E.g., $y$ is salary, $x_3$ tells whether $i$ works or not, $x_2$ is a dummy for training, and $x_1$ a bias:

```math
(y_i, \mathbf{x}_i) = (y_i, 1, x_{2i}, x_{3i}).
```

Split the set of $n$ individuals $I$ into two subsets $I_U$ and $I_L$ where indices "U"  and "L" stand for "unlabelled" and "labeled". Without loss of generality, $I_U\equiv\lbrace 1,\ldots,m\rbrace$, $m<n$, and $I_L\equiv I\setminus I_U$. 

### Data

Tan Tran, Evens Salies. How important is open-source science for invention speed. 2023. [hal-04239561](https://sciencespo.hal.science/hal-04239561v1)

### Bibliography

[^1]:  Géron, A., 2022. _Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow_, O'Reilly, 834 p.
