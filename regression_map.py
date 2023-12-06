# Géron (2022)
#   Ch. 2, section "Instance-based versus model-based learning"

from    pathlib import Path
import  pandas as pd
import  tarfile         # Library to dezip .tgz files
import  urllib.request
import  matplotlib.pyplot as plt

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "http://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()                       # Token for dataframe

# Simple descriptions, statistics about the data

housing.info()                                      # A few stat about the file (nb. obs., nb. variables, ...)
print(housing.head())                               # print() is not in the original code
print(housing["ocean_proximity"].value_counts())    # Différentes modalités de ocean_proximity
print(housing.describe())                           # Stat de base des variables

# Plot an histogram for each variable in a 3 x 3 matrix

housing.hist(bins=50, figsize=(12, 8))
plt.show()

# Create a test set