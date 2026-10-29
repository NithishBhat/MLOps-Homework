# Slim Python base image (the original lab used the full python:3.10 image, which is much larger)
FROM python:3.11-slim

# No .pyc files, unbuffered logs (so docker logs shows output right away), no pip cache in the image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OUTPUT_DIR=/app/output

# Set the working directory in the container
WORKDIR /app

# Install dependencies before copying the code, so this layer is reused when only the code changes
COPY src/requirements.txt .
RUN pip install -r requirements.txt

# Copy the training and prediction scripts
COPY src/main.py src/predict.py ./

# Run as a non-root user that owns the output folder
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p "$OUTPUT_DIR" \
    && chown appuser "$OUTPUT_DIR"
USER appuser

# Models and metrics are written here; mount a host folder to keep them
VOLUME ["/app/output"]

# Train and compare all models by default (override with -e MODEL=... or a different command)
CMD ["python", "main.py"]
