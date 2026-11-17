"""The 13 measurements the wine API expects, with slider ranges taken from the Wine dataset."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    name: str      # field name in the API request
    label: str
    unit: str
    min: float     # dataset minimum
    max: float     # dataset maximum
    median: float  # dataset median, used as the default
    step: float


FEATURES = [
    Feature("alcohol", "Alcohol", "% vol", 11.03, 14.83, 13.05, 0.01),
    Feature("malic_acid", "Malic acid", "g/L", 0.74, 5.80, 1.87, 0.01),
    Feature("ash", "Ash", "g/L", 1.36, 3.23, 2.36, 0.01),
    Feature("alcalinity_of_ash", "Alcalinity of ash", "", 10.6, 30.0, 19.5, 0.1),
    Feature("magnesium", "Magnesium", "mg/L", 70.0, 162.0, 98.0, 1.0),
    Feature("total_phenols", "Total phenols", "", 0.98, 3.88, 2.36, 0.01),
    Feature("flavanoids", "Flavanoids", "", 0.34, 5.08, 2.14, 0.01),
    Feature("nonflavanoid_phenols", "Nonflavanoid phenols", "", 0.13, 0.66, 0.34, 0.01),
    Feature("proanthocyanins", "Proanthocyanins", "", 0.41, 3.58, 1.56, 0.01),
    Feature("color_intensity", "Color intensity", "", 1.28, 13.0, 4.69, 0.01),
    Feature("hue", "Hue", "", 0.48, 1.71, 0.97, 0.01),
    Feature("od280_od315", "OD280/OD315 of diluted wines", "", 1.27, 4.0, 2.78, 0.01),
    Feature("proline", "Proline", "mg/L", 278.0, 1680.0, 673.5, 1.0),
]
FEATURE_NAMES = [f.name for f in FEATURES]

CULTIVAR_LABELS = {"cultivar_1": "Cultivar 1", "cultivar_2": "Cultivar 2", "cultivar_3": "Cultivar 3"}

# Starting points for the sliders: the dataset median, and one real wine of each cultivar
PRESETS = {
    "Dataset median": {f.name: f.median for f in FEATURES},
    "Cultivar 1 wine (dataset row 0)": {
        "alcohol": 14.23, "malic_acid": 1.71, "ash": 2.43, "alcalinity_of_ash": 15.6, "magnesium": 127.0,
        "total_phenols": 2.8, "flavanoids": 3.06, "nonflavanoid_phenols": 0.28, "proanthocyanins": 2.29,
        "color_intensity": 5.64, "hue": 1.04, "od280_od315": 3.92, "proline": 1065.0,
    },
    "Cultivar 2 wine (dataset row 100)": {
        "alcohol": 12.08, "malic_acid": 2.08, "ash": 1.7, "alcalinity_of_ash": 17.5, "magnesium": 97.0,
        "total_phenols": 2.23, "flavanoids": 2.17, "nonflavanoid_phenols": 0.26, "proanthocyanins": 1.4,
        "color_intensity": 3.3, "hue": 1.27, "od280_od315": 2.96, "proline": 710.0,
    },
    "Cultivar 3 wine (dataset row 170)": {
        "alcohol": 12.2, "malic_acid": 3.03, "ash": 2.32, "alcalinity_of_ash": 19.0, "magnesium": 96.0,
        "total_phenols": 1.25, "flavanoids": 0.49, "nonflavanoid_phenols": 0.4, "proanthocyanins": 0.73,
        "color_intensity": 5.5, "hue": 0.66, "od280_od315": 1.83, "proline": 510.0,
    },
}
