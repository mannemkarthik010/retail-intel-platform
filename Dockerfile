FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Bake the pipeline artifacts into the image at build time (synthetic data
# generation + rolling-origin backtest + anomaly scan + forward forecast +
# monitoring simulation) so the container starts instantly and every image
# is a reproducible, versioned snapshot -- consistent with the offline/
# online split described in docs/ARCHITECTURE.md.
RUN python data/generate_data.py \
    && python scripts/run_pipeline.py \
    && python scripts/run_monitoring_sim.py

EXPOSE 8000
CMD ["python", "app/server.py"]
