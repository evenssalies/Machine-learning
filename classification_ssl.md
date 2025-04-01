## Classification - Semi-Supervised learning (SSL)[^1]

### Evens Salies, v1: 01/2025

In ML, the concept of SSL is used in situations where, for some $i$'s, $Y_i$ is not observed. In econometrics, such a situation occurs e.g. when $Y_i$ is missing at random (MAR). But also, when there is sample selection. Imagine an innovation problem: $Y$ is sales, $X$ tells whether firm $i$ has transformed R&D into some marketed product, $D$ is a dummy that takes the value 1 if $i$ has received some subsidy:

```math
(Y_i, D_i, X_i).
\\[12pt]
```

We want to evaluate $D\rightarrow Y$. This is impossible for firms that have not completed their R&D program, regardless whether they received the subsidy or not. ML does not raise sample selection problems like this one. As stated in Géron (2022, p. 14), "you will often have plenty of unlabeled instances [$Y_i$ not observed], and few labeled instances." Split the set of $n$ firms $I$ into two subsets $I_U$ and $I_L$ where indices "U"  and "L" stand for "unlabelled" and "labeled". Without loss of generality, $I=I_U+I_L$, and $I_U\cap I_L\neq\emptyset$. 

### Data

Tan Tran, Evens Salies. How important is open-source science for invention speed. 2023. [hal-04239561](https://sciencespo.hal.science/hal-04239561v1)

### Bibliography

[^1]:  Géron, A., 2022. _Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow_, O'Reilly, 834 p.
