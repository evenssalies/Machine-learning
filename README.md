## What's in here?[^1]

(1) You'll find material on Machine learning (ML) for problems where one's aim is to measure the effects of some more or less controlled $D$ on some $Y$ in the presence of confounders $X$. ML techniques will be discussed *as I learn them*, with as objective to show when they can be useful to causal inference. (2) In the different code comments I pay particular attention to differences between ML and Econometrics from the perspective of causal inference. To put it more clearly, my aim is not to compare ML and Econometrics for prediction $E(Y|X)$, but to see how ML methods can be used in Econometrics for estimating causal effects $E(Y(1)-Y(0)|D(X))$, where $D$ may be continuous. (3) You'll also find critical references on how AI is changing our perception of science, and the way we live in societies.

Regarding (1), I rely on:

- Géron, A., 2022. _Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow_, O'Reilly, which applies instance- and model-based learning methods to economic and other data.
- [Machine learning in Python with scikit-learn](https://www.fun-mooc.fr/fr/cours/machine-learning-python-scikit-learn/), FUN-MOOC.

As to point (2), I'll manage to build some kind of dictionary within which one can find words from ML and their translation into econometrics. A few examples: (i) features, factors, explanatory variables are the same thing, (ii) the method of __Least Squares__ used in _Econometrics_ to estimate a model parameters is used for __training__ in ML, (iii) the __sum of squared residuals__ is the __cost function__, (iv) __Overfitting__ relates to what econometricians call __model saturation__ when specifying and econometric model. A saturated model predicts very well. But saturing a model, though it can be a good control technique, is not key in causal inference. Causal inference requires assumptions and relies on matching techniques where ML can be useful. Some references may help to link the jargon and methods of ML and eocnometrics:

- Athey, S., Imbens, G., 2019. _Machine learning methods that economists should know about_,  [Lien](https://www.annualreviews.org/doi/10.1146/annurev-economics-080217-053433 "Athey, S., Imbens, G. (2019)").

I firmly believe there are plenty of econometric methods that data scientists should know about. Before to merge the two disciplines, first I need to learn ML methods, which will take a few years.

On (3):

- Le Cun, Y., 2014. _Quand la machine apprend_, Odile Jacob.
- Mallat, S., 2018. _Sciences des données et apprentissage en grande dimension_ &ndash; Leçons inaugurales du Collège de France, fayard.
- Sadin, E., 2023. _La vie spectrale_ &ndash; Penser l'ère du métavers et des IA génératives, Grasset.

## Programs

- Perceptron (binary classification)
- Linear regression with a test set
- ...

[^1]: __Acknowledgments.__ I am grateful to Alexandre Mutel.
