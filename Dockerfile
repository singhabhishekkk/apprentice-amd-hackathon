# Router + demo container (CPU): serves the Apprentice router that fronts the
# vLLM instance on the AMD pod and falls back to Fireworks AI.
FROM python:3.12-slim

WORKDIR /app
COPY requirements-router.txt .
RUN pip install --no-cache-dir -r requirements-router.txt

COPY router/ router/

EXPOSE 8900
CMD ["uvicorn", "router.main:app", "--host", "0.0.0.0", "--port", "8900"]
