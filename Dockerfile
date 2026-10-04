# EchoMatch — production image
#
# Build:  docker build -t echomatch .
# Run:    docker run -p 8000:8000 echomatch
# Then open http://localhost:8000

FROM python:3.11-slim

# System deps: a C/C++ toolchain is needed to build python-Levenshtein and
# some LightGBM/gensim wheels on platforms without prebuilt wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${PORT}/healthz', timeout=3)" || exit 1

# 2 workers x 2 threads is a reasonable default for a model-serving Flask
# app on a small instance; tune --workers to available CPU cores and
# --timeout upward if cold-start model loading is slow.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 2 --threads 2 --timeout 60 app:app"]
