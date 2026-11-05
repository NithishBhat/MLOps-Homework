"""Request and response models for the API."""
from pydantic import BaseModel, ConfigDict, Field

# First wine in the dataset (a cultivar_1 wine), shown as the example in /docs
EXAMPLE_WINE = {
    "alcohol": 14.23,
    "malic_acid": 1.71,
    "ash": 2.43,
    "alcalinity_of_ash": 15.6,
    "magnesium": 127,
    "total_phenols": 2.8,
    "flavanoids": 3.06,
    "nonflavanoid_phenols": 0.28,
    "proanthocyanins": 2.29,
    "color_intensity": 5.64,
    "hue": 1.04,
    "od280_od315": 3.92,
    "proline": 1065,
}


class WineFeatures(BaseModel):
    """Chemical analysis of one wine. Every measurement must be a positive number."""

    # Reject unknown fields, so a typo like "alchol" is an error instead of being ignored
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_WINE]})

    alcohol: float = Field(gt=0, le=100, description="Alcohol (% by volume)")
    malic_acid: float = Field(gt=0, description="Malic acid (g/L)")
    ash: float = Field(gt=0, description="Ash (g/L)")
    alcalinity_of_ash: float = Field(gt=0, description="Alcalinity of ash")
    magnesium: float = Field(gt=0, description="Magnesium (mg/L)")
    total_phenols: float = Field(gt=0, description="Total phenols")
    flavanoids: float = Field(gt=0, description="Flavanoids")
    nonflavanoid_phenols: float = Field(gt=0, description="Nonflavanoid phenols")
    proanthocyanins: float = Field(gt=0, description="Proanthocyanins")
    color_intensity: float = Field(gt=0, description="Color intensity")
    hue: float = Field(gt=0, description="Hue")
    od280_od315: float = Field(gt=0, description="OD280/OD315 of diluted wines")
    proline: float = Field(gt=0, description="Proline (mg/L)")


class BatchRequest(BaseModel):
    wines: list[WineFeatures] = Field(min_length=1, max_length=100)


class Prediction(BaseModel):
    cultivar: str
    class_id: int
    confidence: float = Field(description="Probability of the predicted cultivar")
    probabilities: dict[str, float]


class PredictionResponse(Prediction):
    model_version: str


class BatchResponse(BaseModel):
    model_version: str
    count: int
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None


class ModelInfo(BaseModel):
    version: str
    trained_at: str
    model: str
    n_estimators: int
    seed: int
    n_train: int
    n_test: int
    metrics: dict[str, float]
    features: list[str]
    classes: list[str]
    feature_importance: dict[str, float]
    sklearn_version: str
