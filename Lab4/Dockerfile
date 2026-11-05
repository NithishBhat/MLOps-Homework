# Slim Python base image
FROM python:3.11-slim

# No .pyc files, unbuffered logs, no pip cache in the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies before copying the code, so this layer is reused when only the code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/

# Train at build time, so the image ships with a model and serves as soon as it starts
RUN python -m src.train

# Run the server as a non-root user
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000

# Docker marks the container unhealthy if /health stops answering
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

# No --reload here: that's only for development
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
