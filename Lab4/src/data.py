"""Wine dataset: 178 wines from three cultivars, 13 chemical measurements each."""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# API field names, in the column order the model is trained on. sklearn calls the
# 12th column "od280/od315_of_diluted_wines", which isn't a valid field name.
FEATURES = [
    "alcohol",
    "malic_acid",
    "ash",
    "alcalinity_of_ash",
    "magnesium",
    "total_phenols",
    "flavanoids",
    "nonflavanoid_phenols",
    "proanthocyanins",
    "color_intensity",
    "hue",
    "od280_od315",
    "proline",
]
CLASS_NAMES = ["cultivar_1", "cultivar_2", "cultivar_3"]


def load_data():
    wine = load_wine()
    return wine.data, wine.target


def split_data(X, y, test_size=0.25, seed=42):
    """Stratified split, so all three cultivars appear in the test set in the same proportions."""
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
