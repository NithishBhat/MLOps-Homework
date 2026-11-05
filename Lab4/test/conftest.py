import pytest
from fastapi.testclient import TestClient

from src import train


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("model")
    train.main(["--model-dir", str(path), "--n-estimators", "50"])
    return path


@pytest.fixture
def client(model_dir, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(model_dir))
    from src.main import app

    # Using the client as a context manager runs the app's startup, which loads the model
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path))
    from src.main import app

    with TestClient(app) as c:
        yield c
