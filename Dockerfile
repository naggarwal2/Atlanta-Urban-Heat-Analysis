FROM python:3.12-slim

WORKDIR /app

# System libraries required for geopandas / pyogrio spatial boundaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY sar_model_artifact.joblib api.py app.py "Official Atlanta Neighborhood Boundaries.geojson" ./

EXPOSE 8080 8501

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]