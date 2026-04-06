"""
prepare_data.py
================
Converts raw Recipe1MSubs CSV files into vocab.json + split JSONs for train.py.

Usage:
  python train/prepare_data.py \
    --subs_train data/raw/train.csv \
    --subs_val   data/raw/val.csv   \
    --subs_test  data/raw/test.csv  \
    --out_dir    data/processed

Expected CSV columns: recipe_id, ingredient_list (semicolon-separated), source, target
"""
import argparse, csv, json, random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--subs_train", required=True)
    p.add_argument("--subs_val",   required=True)
    p.add_argument("--subs_test",  required=True)
    p.add_argument("--out_dir",    default="data/processed")
    p.add_argument("--prod_frac",  type=float, default=0.5)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def load_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "recipe_id":          row.get("recipe_id", ""),
                "recipe_ingredients": [i.strip().lower() for i in row.get("ingredient_list","").split(";")],
                "source":             row["source"].strip().lower(),
                "target":             row["target"].strip().lower(),
            })
    return rows


def build_vocab(all_samples):
    ings = set()
    for s in all_samples:
        ings.update(s["recipe_ingredients"])
        ings.add(s["source"]); ings.add(s["target"])
    return {ing: i for i, ing in enumerate(sorted(ings))}


def encode(samples, vocab):
    out = []
    for s in samples:
        if s["source"] not in vocab or s["target"] not in vocab:
            continue
        out.append({
            "recipe_id":          s["recipe_id"],
            "recipe_ingredients": [vocab[i] for i in s["recipe_ingredients"] if i in vocab],
            "source":             vocab[s["source"]],
            "target":             vocab[s["target"]],
        })
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    train_raw = load_csv(args.subs_train)
    val_raw   = load_csv(args.subs_val)
    test_raw  = load_csv(args.subs_test)
    print(f"Raw: train={len(train_raw)}, val={len(val_raw)}, test={len(test_raw)}")

    vocab = build_vocab(train_raw + val_raw + test_raw)
    print(f"Vocab size: {len(vocab)}")
    with open(out / "vocab.json", "w") as f:
        json.dump(vocab, f)

    test_enc = encode(test_raw, vocab)
    random.shuffle(test_enc)
    split = int(len(test_enc) * (1 - args.prod_frac))

    splits = {
        "train":      encode(train_raw, vocab),
        "val":        encode(val_raw,   vocab),
        "test":       test_enc[:split],
        "production": test_enc[split:],
    }
    for name, data in splits.items():
        p = out / f"{name}.json"
        with open(p, "w") as f:
            json.dump(data, f)
        print(f"  {name}: {len(data)} samples → {p}")

    print("Done.")


if __name__ == "__main__":
    main()
