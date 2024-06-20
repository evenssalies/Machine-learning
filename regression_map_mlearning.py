import  matplotlib.pyplot as plt    # Graphiques
import  pandas as pd                # Fonctions de base
import  os as os                    # Commandes DOS

# Read the CSV file (from the working directory!)
housing = pd.read_csv("datasets\housing\housing.csv")

# Simple descriptions, statistics about the data
#   Top five rows (print() is not in the original code)
print(housing.head())                               

#   Stats about the file (nb. obs., nb. variables, ...)
housing.info()                                     

#   Différentes modalités de "ocean_proximity"
print(housing["ocean_proximity"].value_counts())   

#   Stat de base des variables continues
print(housing.describe())                           

# Histogramme des variables continues
housing.hist(bins=50, figsize=(12, 8))
plt.show()

# Crée plusieurs sous-échantillons (stratification). Dans
#  chacun, on sépare les sous-ensembles test-entraînement.
import  numpy as np

#   Coarsening de la variable "median_income"
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
                               labels=[1, 2, 3, 4, 5])

#   Histogramme des catégories
housing["income_cat"].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel("Income category")
plt.ylabel("Fréquence")
plt.show()

#   Sépare les sous-ensembles test-entraînement  
from sklearn.model_selection import train_test_split
strat_train_set, strat_test_set = train_test_split(
    housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

#   Efface l'écran avant
os.system('cls')

#   Visualise ces sous-ensembles (vérifie le split 0.2-0.8)
strat_train_set.info()
strat_test_set.info()

#   Visualise la fréquence relative de "income_cat" dans chaque
#       sous-ensemble et dans les données de départ
os.system('cls')
strat_train_set["income_cat"].value_counts()/len(strat_train_set)
strat_test_set["income_cat"].value_counts()/len(strat_test_set)
housing["income_cat"].value_counts()/len(housing)

# Virer "income_cat" des ensembles d'entraînement et test
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
housing.drop("income_cat", axis=1, inplace=True)    # Pas sur qu'il faille le faire pour housing

# Etudier l'ensemble d'entraînement