# Official Airflow image. It already has numpy, scipy, scikit-learn and joblib.
FROM apache/airflow:3.3.2-python3.12

# Bake the extra package into the image instead of installing it on every start
# (the original lab used _PIP_ADDITIONAL_REQUIREMENTS, which reinstalls each time the container starts).
# Pinning apache-airflow stops pip from upgrading or downgrading Airflow itself.
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
