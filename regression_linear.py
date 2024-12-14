# Scikit-learn: méthodes fit(), predict(), score()
#  
# Géron (2022, pp.25-26)
#   Ch. 1, section "Instance-based versus model-based learning"
#   Data : "lifesat.csv"
# Machine learning in Python with scikit-learn
#   Adaptation of code source: https://www.fun-mooc.fr/fr/cours/machine-learning-python-scikit-learn/
#   Data: 1590 OpenML dataset: Ron Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid",
#       Proceedings of the Second International Conference on Knowledge Discovery and Data Mining, 1996

import  matplotlib.pyplot as plt
import  numpy as np
import  pandas as pd
import  os as os

from    sklearn.linear_model import LinearRegression

# First example: linear regression
# 
# Download and prepare the data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")

# Handle missing values
lifesat = lifesat.dropna()

X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

# Visualize the data
lifesat.plot(kind='scatter', grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Cyprus
#   Cyprus' GDP per capita in 2020
X_new = [[37_655.2]]
print(model.predict(X_new))

# KNeighborsRegressor: instance-based learning (k-nearest neighbors) 
from    sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=3)

# fit method: Train the new model
model.fit(X, y)

# predict method: Make a prediction for Cyprus
#   Cyprus' GDP per capita in 2020
X_new = [[37_655.2]]
print(model.predict(X_new))

# Second example: classification
#
# KNeighborsClassifier 
# Va chercher la prochaine base dans openml; voir print.py
import openml
dataset = openml.datasets.get_dataset(1590)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Vire les missing (ou logique)
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

# Classification KNN
#  Sans échantillon test: on utilise les mêmes données pour l'apprentissage et la prédiction,
#                         donc pour l'évaluation du modèle
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

# Classification KNN
#  Avec échantillon test: généralisation du modèle
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)

# Vire les missing et split les données en échantillons test-entraînement
Xy = pd.concat([X, y], axis=1)
Xy = Xy.dropna()
from sklearn.model_selection import train_test_split
Xy_train, Xy_test = train_test_split(Xy, test_size=0.2, random_state=42)

# Split back X et y
os.system('cls')
X_test = Xy_test.drop(columns=[dataset.default_target_attribute])
y_test = Xy_test[dataset.default_target_attribute]
X_test = X_test[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]
X_train = Xy_train.drop(columns=[dataset.default_target_attribute])
y_train = Xy_train[dataset.default_target_attribute]
X_train = X_train[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]

# Vérifie les dimensions des tableaux
print(f"\nX_test a {X_test.shape[0]} observations et {X_test.shape[1]} variables.")
print(f"X_test a {X_train.shape[0]} observations et {X_train.shape[1]} variables.\n")

# Classification KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()

# On a juste besoin d'entrainer le modèle
_ = model.fit(X_train, y_train)

# Score du modèle : taux de succès moyen ou performance de généralisation/prédictive/statistique
accuracy = model.score(X_test, y_test)
model_name = model.__class__.__name__
print(f"La qualité du modèle {model_name} est de {accuracy:.3f}")