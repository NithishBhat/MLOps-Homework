import pytest
import requests

import api


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def respond_with(monkeypatch, response=None, error=None):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        if error:
            raise error
        return response

    monkeypatch.setattr(requests, "request", fake_request)
    return calls


def test_predict_posts_json_with_timeout(monkeypatch):
    calls = respond_with(monkeypatch, FakeResponse(200, {"cultivar": "cultivar_1"}))
    assert api.predict({"alcohol": 13.0}) == {"cultivar": "cultivar_1"}
    method, url, kwargs = calls[0]
    assert (method, url) == ("POST", f"{api.API_URL}/predict")
    assert kwargs["json"] == {"alcohol": 13.0}
    assert kwargs["timeout"] == api.TIMEOUT


def test_batch_wraps_wines(monkeypatch):
    calls = respond_with(monkeypatch, FakeResponse(200, {"count": 2}))
    api.predict_batch([{"a": 1}, {"a": 2}])
    assert calls[0][1].endswith("/predict/batch")
    assert calls[0][2]["json"] == {"wines": [{"a": 1}, {"a": 2}]}


def test_422_becomes_readable_messages(monkeypatch):
    body = {"detail": [
        {"loc": ["body", "proline"], "msg": "Field required"},
        {"loc": ["body", "alcohol"], "msg": "Input should be greater than 0"},
    ]}
    respond_with(monkeypatch, FakeResponse(422, body))
    with pytest.raises(api.ApiError) as err:
        api.predict({})
    assert err.value.status == 422
    assert err.value.details == ["proline: Field required", "alcohol: Input should be greater than 0"]


def test_503_shows_the_api_detail(monkeypatch):
    respond_with(monkeypatch, FakeResponse(503, {"detail": "Model not loaded."}))
    with pytest.raises(api.ApiError, match="API error 503: Model not loaded."):
        api.model_info()


def test_non_json_error_body(monkeypatch):
    respond_with(monkeypatch, FakeResponse(502, ValueError("not json")))
    with pytest.raises(api.ApiError, match="API error 502"):
        api.health()


def test_connection_error(monkeypatch):
    respond_with(monkeypatch, error=requests.ConnectionError("refused"))
    with pytest.raises(api.ApiError, match="Can't reach the API"):
        api.health()


def test_timeout(monkeypatch):
    respond_with(monkeypatch, error=requests.Timeout())
    with pytest.raises(api.ApiError, match="didn't answer"):
        api.health()
