## What's in here? _(to be polished ever and ever)_[^1]

(1) You'll find material on causal inference for problems where one's aim is to measure the effects of some more or less controlled _D_ on some _Y_ in the presence of confounders _X_. Machine learning (ML) techniques will be discussed as I learn them, with as objective to show when they can be useful to causal inference. (2) In the different code comments I pay particular attention to differences between ML and Econometrics from the perspective of causal inference. To put it more clearly, my aim is not to compare ML and Econometrics for prediction E(_Y_|_X_), but to see how ML methods can be used in Econometrics for estimating causal effects E(_Y_(1)-_Y_(0)|_D_(_X_)), where _D_ may be continuous.

Regarding (1), I rely on:

- the seminal book by Géron, A. (2022) _Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow_, O'Reilly, which applies instance- and model-based learning methods to economic and other data;
- Le Cun, Y. (2014) _Quand la machine apprend_, Odile Jacob;
- [Machine learning in Python with scikit-learn](https://www.fun-mooc.fr/fr/cours/machine-learning-python-scikit-learn/).

As to point (2), I'll manage to build some kind of dictionary within which one can find words from ML and their translation into econometrics. A few examples: (i) features, factors, explanatory variables are the same thing, (ii) the method of __Least Squares__ used in _Econometrics_ to estimate a model parameters is used for __training__ in ML, (iii) the __sum of squared residuals__ is the __cost function__, (iv) __Overfitting__ relates to what econometricians call __model saturation__ when specifying and econometric model. A saturated model predicts very well. But saturing a model, though it can be a good control technique, is not key in causal inference. Causal inference requires assumptions and relies on matching techniques where ML can be useful. Some references may help to link the jargon and methods of ML and eocnometrics:

- Athey, S. and Imbens, G.'s (2019) paper: _Machine learning methods that economists should know about_,  [Lien](https://www.annualreviews.org/doi/10.1146/annurev-economics-080217-053433 "Athey, S., Imbens, G. (2019)").

I firmly believe there are plenty of econometric methods that data scientists should know about. Before to merge the two disciplines, first I need to learn ML methods, which will take a few years.

## Plan _(building ...)_

Program 1.
Program 2.

...

[^1]: __Acknowledgments.__ I am grateful to Alexandre Mutel.
