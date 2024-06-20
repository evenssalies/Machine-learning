# Géron (2022)
#   Ch. 2, section "Instance-based versus model-based learning"
#   Import the dataset

from    pathlib import Path
import  pandas as pd
import  tarfile         # Library to dezip .tgz files
import  urllib.request

# Fonction qui charge et décompresse les données

def load_housing_data():
    tarball_path = Path("datasets/housing.tgz")

# Si les données ne sont pas déjà dans le dossier
    if not tarball_path.is_file():

# Crée le dossier `datasets'
        Path("datasets").mkdir(parents=True, exist_ok=True)

# Va chercher le fichier de données à cette adresse
        url = "http://github.com/ageron/data/raw/main/housing.tgz"
        urllib.request.urlretrieve(url, tarball_path)

# Le décompresse dans le dossier `datasets' du Working Directory
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets")
    return pd.read_csv(Path("datasets/housing/housing.csv"))

housing = load_housing_data()