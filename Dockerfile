FROM python:3.11-slim

WORKDIR /app

# Copie des fichiers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY models/ ./models/

# Port exposé
EXPOSE 8000

# Lancement
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]