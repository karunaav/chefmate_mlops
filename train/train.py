"""
ChefMate Ingredient Substitution — Training Script
===================================================
Supports: baseline (frequency), mlp, gismo

Usage:
  python train/train.py --config train/config_gismo.json
  python train/train.py --model baseline --data_dir data/processed
  python train/train.py --model gismo --embed_dim 256 --lr 0.0005
"""

import argparse, json, os, time, subprocess, logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# DEFAULT CONFIG
# ──────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "model":                "gismo",      # baseline | mlp | gismo
    "data_dir":             "data/processed",
    "embed_dim":            128,
    "hidden_dim":           256,
    "num_layers":           2,
    "dropout":              0.3,
    "lr":                   1e-3,
    "weight_decay":         1e-5,
    "batch_size":           256,
    "epochs":               30,
    "patience":             5,
    "top_k":                [1, 5, 10],
    "seed":                 42,
    "mlflow_tracking_uri":  "http://localhost:5000",
    "experiment_name":      "chefmate-substitution",
    "run_name":             None,
}

# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class SubstitutionDataset(Dataset):
    """
    Each JSON record:
      { "recipe_ingredients": [int, ...], "source": int, "target": int }
    """
    def __init__(self, path, vocab_size):
        with open(path) as f:
            self.samples = json.load(f)
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ctx = torch.zeros(self.vocab_size)
        for i in s["recipe_ingredients"]:
            ctx[i] = 1.0
        return ctx, torch.tensor(s["source"], dtype=torch.long), \
                    torch.tensor(s["target"], dtype=torch.long)


def load_vocab(data_dir):
    with open(Path(data_dir) / "vocab.json") as f:
        return json.load(f)


def build_loaders(cfg, vocab_size):
    d = Path(cfg["data_dir"])
    def make(split):
        ds = SubstitutionDataset(d / f"{split}.json", vocab_size)
        return DataLoader(ds, batch_size=cfg["batch_size"],
                          shuffle=(split == "train"),
                          num_workers=4, pin_memory=True)
    return make("train"), make("val"), make("test")

# ──────────────────────────────────────────────────────────────
# MODELS
# ──────────────────────────────────────────────────────────────
class FrequencyBaseline:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.counts = defaultdict(lambda: defaultdict(int))

    def fit(self, train_path):
        with open(train_path) as f:
            for s in json.load(f):
                self.counts[s["source"]][s["target"]] += 1
        log.info("Baseline fitted.")

    def predict(self, source_ids):
        scores = np.zeros((len(source_ids), self.vocab_size))
        for i, src in enumerate(source_ids.tolist()):
            for tgt, cnt in self.counts[src].items():
                scores[i, tgt] = cnt
        return torch.tensor(scores, dtype=torch.float)


class GISMoModel(nn.Module):
    """Source embedding + recipe bag-of-words context → ranked targets."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        layers, in_dim = [], vocab_size
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        self.ctx_enc = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(embed_dim + hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, ctx, src):
        return self.head(torch.cat([self.embed(src), self.ctx_enc(ctx)], dim=-1))


class MLPModel(nn.Module):
    """Simple source-only MLP. No recipe context."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        layers, in_dim = [], embed_dim
        for _ in range(num_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, vocab_size))
        self.net = nn.Sequential(*layers)

    def forward(self, ctx, src):
        return self.net(self.embed(src))

# ──────────────────────────────────────────────────────────────
# METRICS
# ──────────────────────────────────────────────────────────────
def hit_at_k(logits, targets, k_list):
    _, topk = logits.topk(max(k_list), dim=-1)
    return {k: (topk[:, :k] == targets.unsqueeze(1)).any(1).float().mean().item()
            for k in k_list}

# ──────────────────────────────────────────────────────────────
# TRAIN / EVAL
# ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, device):
    model.train()
    loss_sum, n, t0 = 0.0, 0, time.time()
    for ctx, src, tgt in loader:
        ctx, src, tgt = ctx.to(device), src.to(device), tgt.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(ctx, src), tgt)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(tgt); n += len(tgt)
    return loss_sum / n, time.time() - t0

@torch.no_grad()
def eval_epoch(model, loader, device, k_list):
    model.eval()
    loss_sum, n = 0.0, 0
    hit_acc = defaultdict(float)
    for ctx, src, tgt in loader:
        ctx, src, tgt = ctx.to(device), src.to(device), tgt.to(device)
        logits = model(ctx, src)
        loss_sum += F.cross_entropy(logits, tgt).item() * len(tgt); n += len(tgt)
        for k, v in hit_at_k(logits, tgt, k_list).items():
            hit_acc[k] += v * len(tgt)
    return loss_sum / n, {k: hit_acc[k] / n for k in k_list}

@torch.no_grad()
def eval_baseline(baseline, loader, device, k_list):
    hit_acc, n = defaultdict(float), 0
    for _, src, tgt in loader:
        scores = baseline.predict(src).to(device)
        for k, v in hit_at_k(scores, tgt.to(device), k_list).items():
            hit_acc[k] += v * len(tgt)
        n += len(tgt)
    return {k: hit_acc[k] / n for k in k_list}

# ──────────────────────────────────────────────────────────────
# GPU INFO
# ──────────────────────────────────────────────────────────────
def gpu_info():
    info = {"cuda_available": torch.cuda.is_available(),
            "device_count":   torch.cuda.device_count()}
    if torch.cuda.is_available():
        info["device_name"] = torch.cuda.get_device_name(0)
    try:
        out = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=10)
        info["rocm_smi"] = out.stdout[:400]
    except Exception:
        pass
    return info

# ──────────────────────────────────────────────────────────────
# CLI / CONFIG
# ──────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str)
    for k in ["model","data_dir","run_name","mlflow_tracking_uri","experiment_name"]:
        p.add_argument(f"--{k}", type=str)
    for k in ["embed_dim","hidden_dim","num_layers","batch_size","epochs","patience","seed"]:
        p.add_argument(f"--{k}", type=int)
    for k in ["lr","weight_decay","dropout"]:
        p.add_argument(f"--{k}", type=float)
    return p.parse_args()

def build_config(args):
    cfg = dict(DEFAULT_CONFIG)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    for k, v in vars(args).items():
        if v is not None and k != "config":
            cfg[k] = v
    return cfg

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = build_config(args)
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    vocab      = load_vocab(cfg["data_dir"])
    vocab_size = len(vocab)
    log.info("Vocab size: %d", vocab_size)

    mlflow.set_tracking_uri(cfg["mlflow_tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    run_name = cfg["run_name"] or f"{cfg['model']}-{int(time.time())}"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: v for k, v in cfg.items() if not isinstance(v, list)})
        mlflow.log_param("top_k", str(cfg["top_k"]))
        mlflow.log_params({f"gpu_{k}": str(v) for k, v in gpu_info().items()})

        t_total = time.time()

        # ── BASELINE ────────────────────────────────────────────
        if cfg["model"] == "baseline":
            baseline = FrequencyBaseline(vocab_size)
            baseline.fit(Path(cfg["data_dir"]) / "train.json")
            _, val_loader, test_loader = build_loaders(cfg, vocab_size)
            for split, loader in [("val", val_loader), ("test", test_loader)]:
                hits = eval_baseline(baseline, loader, device, cfg["top_k"])
                for k, v in hits.items():
                    mlflow.log_metric(f"{split}_hit@{k}", v)
                log.info("%s hits: %s", split, hits)
            mlflow.log_metric("total_training_time_sec", time.time() - t_total)
            return

        # ── NEURAL MODELS ───────────────────────────────────────
        train_loader, val_loader, test_loader = build_loaders(cfg, vocab_size)

        model = {
            "gismo": GISMoModel,
            "mlp":   MLPModel,
        }[cfg["model"]](vocab_size, cfg["embed_dim"], cfg["hidden_dim"],
                        cfg["num_layers"], cfg["dropout"]).to(device)

        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=cfg["lr"], weight_decay=cfg["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=2, factor=0.5)

        best_hit1, patience_cnt = -1, 0
        ckpt = f"/tmp/best_{run_name}.pt"

        for epoch in range(1, cfg["epochs"] + 1):
            train_loss, ep_sec = train_epoch(model, train_loader, optimizer, device)
            val_loss, val_hits = eval_epoch(model, val_loader, device, cfg["top_k"])
            scheduler.step(val_loss)

            mlflow.log_metrics({
                "train_loss":      train_loss,
                "val_loss":        val_loss,
                "epoch_time_sec":  ep_sec,
                **{f"val_hit@{k}": v for k, v in val_hits.items()},
            }, step=epoch)

            log.info("Epoch %02d | train_loss=%.4f | val_loss=%.4f | "
                     "Hit@1=%.4f Hit@5=%.4f | %.1fs",
                     epoch, train_loss, val_loss,
                     val_hits.get(1, 0), val_hits.get(5, 0), ep_sec)

            if val_hits.get(1, 0) > best_hit1:
                best_hit1 = val_hits.get(1, 0)
                patience_cnt = 0
                torch.save(model.state_dict(), ckpt)
            else:
                patience_cnt += 1
                if patience_cnt >= cfg["patience"]:
                    log.info("Early stopping at epoch %d", epoch)
                    break

        model.load_state_dict(torch.load(ckpt))
        _, test_hits = eval_epoch(model, test_loader, device, cfg["top_k"])
        for k, v in test_hits.items():
            mlflow.log_metric(f"test_hit@{k}", v)
        log.info("Test hits: %s", test_hits)

        mlflow.log_metrics({
            "total_training_time_sec": time.time() - t_total,
            "best_val_hit@1":          best_hit1,
        })
        mlflow.pytorch.log_model(model, "model")
        log.info("Done.")

if __name__ == "__main__":
    main()
