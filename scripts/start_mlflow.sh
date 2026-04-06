#!/bin/bash
# scripts/start_mlflow.sh
# Run this on the Chameleon node to expose MLflow to course staff

pip install mlflow -q

mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///$(pwd)/mlflow.db \
  --default-artifact-root $(pwd)/mlflow-artifacts
