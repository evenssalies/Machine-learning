# Scikit-learn: méthodes fit(), predict(), score()
#
# Je distingue les données d'apprentissage de celles test
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

#   Vire les missing et split les données en échantillons test-entraînement
Xy = pd.concat([X, y], axis=1)
Xy = Xy.dropna()
from sklearn.model_selection import train_test_split
Xy_train, Xy_test = train_test_split(Xy, test_size=0.2, random_state=42)

#   Split back X et y
os.system('cls')
X_test = Xy_test.drop(columns=[dataset.default_target_attribute])
y_test = Xy_test[dataset.default_target_attribute]
X_test = X_test[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]
X_train = Xy_train.drop(columns=[dataset.default_target_attribute])
y_train = Xy_train[dataset.default_target_attribute]
X_train = X_train[['age', 'capital-gain', 'capital-loss', 'hours-per-week']]

#   Vérifie les dimensions des tableaux
print(f"\nX_test a {X_test.shape[0]} observations et {X_test.shape[1]} variables.")
print(f"X_train a {X_train.shape[0]} observations et {X_train.shape[1]} variables.\n")

#   Classe
from sklearn.neighbors import KNeighborsClassifier

#   Instanciation du modèle avec le classificateur (KNN) de la classe
model = KNeighborsClassifier()

#   Entrainement du modèle via l'appel de la méthode fit
_ = model.fit(X_train, y_train)

#   Evaluation du modèle : taux de succès moyen ou performance de généralisation/prédictive/statistique
accuracy = model.score(X_test, y_test)
model_name = model.__class__.__name__
print(f"La qualité du modèle {model_name} est de {accuracy:.3f}")

from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

#   Calcul manuel du rappel dans le cas "<=50K"
print(f"TP: {((y_pred == '<=50K') & (y_test == '<=50K')).sum()}")
print(f"FN: {((y_pred == '>50K') & (y_test == '<=50K')).sum()}")
print(f"TP+FN: {(y_test == '<=50K').sum()}")
print(f"Rappel calculé manuellement: {6326 / (6326 + 516):.3f}")

#   Calcul manuel du rappel dans le cas ">50K"
print(f"TP: {((y_pred == '>50K') & (y_test == '>50K')).sum()}")
print(f"FN: {((y_pred == '<=50K') & (y_test == '>50K')).sum()}")
print(f"TP+FN: {(y_test == '>50K').sum()}")
print(f"Rappel calculé manuellement: {940 / (940 + 1263):.3f}")

#   On retrouve ces nombres dans la matrice confusion
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)