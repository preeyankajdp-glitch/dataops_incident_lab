FROM python:3.11-slim

WORKDIR /app

COPY foodops_env /app/foodops_env

RUN pip install --no-cache-dir fastapi uvicorn pydantic

EXPOSE 8000

CMD ["uvicorn", "foodops_env.server:app", "--host", "0.0.0.0", "--port", "8000"]

