#!/bin/bash
# download_data.sh — Download Recipe1MSubs from GISMo (Facebook Research)
# Run on Chameleon node after SSH-ing in.

set -e
mkdir -p data/raw

echo "Cloning GISMo (contains Recipe1MSubs)..."
git clone https://github.com/facebookresearch/gismo.git data/gismo_repo

echo "Copying substitution splits..."
cp data/gismo_repo/data/substitutions_train.json data/raw/train.json
cp data/gismo_repo/data/substitutions_val.json   data/raw/val.json
cp data/gismo_repo/data/substitutions_test.json  data/raw/test.json

echo "Done. Files in data/raw/"
ls -lh data/raw/
