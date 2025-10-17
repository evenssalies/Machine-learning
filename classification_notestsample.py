# Scikit-learn: méthodes fit(), predict()
# 
# Je ne fais pas de distinction apprentissage/test ici, juste pour illustrer fit() et predict()
#
# Géron (2022, pp. 103-174)
#   Ch. 3 "Classification"
#   Data : "lifesat.csv"
# Machine learning in Python with scikit-learn
#   Adaptation of code source: https://www.fun-mooc.fr/fr/cours/machine-learning-python-scikit-learn/
#   Data: 1590 OpenML dataset: Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",
#       Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996

import  os
import  openml
import  pandas as pd

os.system('cls')

# KNeighborsClassifier 
dataset = openml.datasets.get_dataset(1590)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Vire les missing (OU logique)
Xy = pd.concat([X, y], axis=1)
Xy = Xy.dropna()

# Split back
X = Xy.drop(columns=[dataset.default_target_attribute])
y = Xy[dataset.default_target_attribute]

# Sélectionne un sous-ensemble de variables continues de X
#  Les double-crochets permettent de sélectionner plusieurs colonnes et de garder le DataFrame  
X = X[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]

# Target, label, classe, etc., à chaque discipline ses termes
print(y.head())

# Data, features, factors, etc., à chaque discipline ses termes
print(X.head())
print(X.columns)
print(f"X a {X.shape[0]} observations et {X.shape[1]} variables.")

# Classificateur
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
_ = model.fit(X, y)
ypredict = model.predict(X)

# Compare les prédictions sur les 5 premières observations puis en face à face, c'est mieux
print(f"y :\n {y[:5]}\n y prédit :\n {ypredict[:5]}\n\n")
os.system('cls')
print(f"{'y':<10}{'y prédit':<10}")
for j1, j2 in zip(y[:5], ypredict[:5]):
    print(f"{j1:<10}{j2:<10}")

# Vecteur boléen des succès et nombre de succès 
print(y[:5] == ypredict[:5]) 
print((y[:5] == ypredict[:5]).sum()/5)

# Taux de succès moyen dans l'échantillon (average accuracy or success rate)
print((y == ypredict).mean())