FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    API_URL=http://api:8000

WORKDIR /app

# Dependencies first, so code changes don't reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt

# The dashboard (Dashboard.py, pages/, .streamlit/config.toml, sample files)
COPY dashboard/ .

RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8501

# Streamlit's own health endpoint
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=2)"

CMD ["streamlit", "run", "Dashboard.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
