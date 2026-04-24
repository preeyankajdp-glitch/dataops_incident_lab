FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

COPY pyproject.toml README.md openenv.yaml ./
COPY __init__.py models.py solver.py demo_app.py ./
COPY server ./server
COPY data ./data

RUN pip install --no-cache-dir -e .

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "dataops_incident_gym.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
