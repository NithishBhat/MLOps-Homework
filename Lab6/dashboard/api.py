"""Client for the Lab4 wine API. Every call raises ApiError with a readable message on failure."""
import os

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = float(os.environ.get("API_TIMEOUT", "10"))


class ApiError(Exception):
    def __init__(self, message, status=None, details=None):
        super().__init__(message)
        self.status = status
        self.details = details or []


def validation_messages(body):
    """Turn FastAPI's 422 body into lines like 'proline: Field required'."""
    messages = []
    for error in body.get("detail", []):
        loc = [str(part) for part in error.get("loc", []) if part != "body"]
        messages.append(f"{'.'.join(loc) or 'request'}: {error.get('msg', 'invalid')}")
    return messages


def _request(method, path, **kwargs):
    url = f"{API_URL}{path}"
    try:
        response = requests.request(method, url, timeout=TIMEOUT, **kwargs)
    except requests.Timeout:
        raise ApiError(f"The API at {API_URL} didn't answer within {TIMEOUT:g} s.") from None
    except requests.ConnectionError:
        raise ApiError(f"Can't reach the API at {API_URL}. Is it running?") from None

    if response.status_code == 422:
        raise ApiError("The API rejected the input.", 422, validation_messages(response.json()))
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise ApiError(f"API error {response.status_code}: {detail}", response.status_code)
    return response.json()


def health():
    return _request("GET", "/health")


def model_info():
    return _request("GET", "/model/info")


def predict(wine):
    return _request("POST", "/predict", json=wine)


def predict_batch(wines):
    return _request("POST", "/predict/batch", json={"wines": wines})
