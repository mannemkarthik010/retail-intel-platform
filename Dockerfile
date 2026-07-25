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
    && python scripts/run_monitoring_sim.py \
    && python scripts/retrain_flagged.py

EXPOSE 8000

# The Flask dev server is fine for this demo but prints its own "do not use
# in production" warning. For real deployment, swap in gunicorn instead --
# not validated end-to-end in the sandbox this image was built in (pip
# installs beyond a pre-cached set were blocked there), so it's documented
# here rather than silently swapped in unverified:
#   RUN pip install --no-cache-dir gunicorn
#   CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "app.server:app"]
CMD ["python", "app/server.py"]
