## Classification - Semi-Supervised learning (SSL)[^1]

### Evens Salies, v1: 01/2025

In ML, the concept of SSL is used in situations where, for some $i$'s, $Y_i$ is not observed. In econometrics, such a situation occurs when $Y_i$ is missing at random (MAR) or when there is sample selection. E.g., $Y$ is salary, $X$ tells whether $i$ has a job or not, $D$ is a dummy that takes the value 1 if $i$ has followed some training program:

```math
(Y_i, D_i, X_i).
\\[12pt]
```

Split the set of $n$ individuals $I$ into two subsets $I_U$ and $I_L$ where indices "U"  and "L" stand for "unlabelled" and "labeled". Without loss of generality, $I_U\equiv\lbrace 1,\ldots,m\rbrace$, $m<n$, and $I_L\equiv I\setminus I_U$. 

### Data

Tan Tran, Evens Salies. How important is open-source science for invention speed. 2023. [hal-04239561](https://sciencespo.hal.science/hal-04239561v1)

### Bibliography

[^1]:  Géron, A., 2022. _Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow_, O'Reilly, 834 p.
