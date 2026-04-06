FROM python:3.10-slim
WORKDIR /app
COPY serve/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve/app.py .
COPY data/processed/vocab.json ./data/processed/vocab.json
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
