# 🍽️ ChefMate MLOps — Ingredient Substitution

> Given a recipe and a missing ingredient, predict and rank suitable substitutions.  
> Built on top of [Mealie](https://mealie.io/) · Trained on [Recipe1MSubs](https://github.com/facebookresearch/gismo) · Tracked with MLflow · Deployed on Chameleon Cloud (AMD MI100).

---

## 📁 Repository Structure

```
chefmate_mlops/
├── train/                   # Model training (runs on Chameleon GPU node)
│   ├── docker_amd/
│   │   └── Dockerfile       # ROCm-based training container (AMD MI100)
│   ├── train.py             # Single training script — model selected via config
│   ├── prepare_data.py      # Build vocab + train/val/test/production splits
│   ├── config_baseline.json # Frequency baseline config
│   ├── config_mlp.json      # MLP candidate config
│   └── config_gismo.json    # GISMo embedding model config
├── serve/                   # FastAPI inference service
│   ├── Dockerfile
│   ├── app.py               # /substitute endpoint
│   └── requirements.txt
├── data/                    # Data pipeline scripts
│   ├── download_data.sh     # Download Recipe1M + Recipe1MSubs
│   └── validate_splits.py   # Sanity-check data splits
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
├── monitoring/
│   ├── prometheus/
│   │   └── prometheus.yml
│   └── grafana/
│       └── dashboard.json
├── scripts/
│   └── start_mlflow.sh      # Launch MLflow tracking server on Chameleon
├── docker-compose.yml       # Full stack: Mealie + FastAPI + MLflow + Monitoring
├── .github/
│   └── workflows/
│       └── ci.yml
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/chefmate_mlops.git
cd chefmate_mlops
```

### 2. Prepare data (on Chameleon node)
```bash
bash data/download_data.sh
python train/prepare_data.py \
  --subs_train data/raw/train.csv \
  --subs_val   data/raw/val.csv   \
  --subs_test  data/raw/test.csv  \
  --out_dir    data/processed
```

### 3. Build training container
```bash
docker build -t chefmate-train:latest -f train/docker_amd/Dockerfile .
```

### 4. Run a training candidate
```bash
# Baseline
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add $(stat -c "%g" /dev/kfd) \
  --group-add $(stat -c "%g" /dev/dri/card0) \
  --shm-size=12g \
  -v $(pwd):/workspace \
  chefmate-train:latest \
  python train/train.py --config train/config_baseline.json

# GISMo model
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add $(stat -c "%g" /dev/kfd) \
  --group-add $(stat -c "%g" /dev/dri/card0) \
  --shm-size=12g \
  -v $(pwd):/workspace \
  chefmate-train:latest \
  python train/train.py --config train/config_gismo.json
```

### 5. Start the full stack
```bash
docker compose up -d
```

Services started:
| Service | URL |
|---|---|
| Mealie | http://localhost:9000 |
| ChefMate API | http://localhost:8000 |
| MLflow | http://localhost:5000 |
| Grafana | http://localhost:3000 |
| Prometheus | http://localhost:9090 |

---

## 🧠 Models & Candidates

| Candidate | Config | Notes |
|---|---|---|
| Frequency Baseline | `config_baseline.json` | No training; ranks by co-occurrence count |
| MLP | `config_mlp.json` | Source embedding → target distribution |
| GISMo-style | `config_gismo.json` | Source + recipe context → ranked targets |

Evaluation metric: **Hit@K** — fraction of queries where the correct substitution appears in the top K predictions.

---

## 📊 MLflow

All training runs are tracked at the MLflow server running on Chameleon.  
Each run logs:
- **Config params**: model type, embed_dim, lr, batch_size, etc.
- **Quality metrics**: Hit@1, Hit@5, Hit@10 (val + test)
- **Cost metrics**: epoch time (sec), total training time (sec)
- **Environment**: GPU name, ROCm version, CUDA availability

---

## 👥 Team — Project 01

