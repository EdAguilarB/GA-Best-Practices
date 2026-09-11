#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ">>> Installing dependencies (may take a few minutes on first run)..."
uv sync
source "$REPO/.venv/bin/activate"

mkdir -p "$REPO/initial_randstates" "$REPO/last_gen_params" "$REPO/rand_states" \
         "$REPO/quick_files" "$REPO/full_files"

echo ">>> Loading component CSVs and reference features..."
cd "$REPO/GA_code"
echo ">>> Running GA (pop=8, generations=3)..."
python GA_main.py \
    pop_size 8 opt_bg TEST \
    --n_generations 3 \
    --maximize \
    --aldehydes   "$REPO/aldehydes_curated_v3.csv" \
    --acids       "$REPO/acids_curated_v3.csv" \
    --amines      "$REPO/amines_curated_v3.csv" \
    --isocyanides "$REPO/isocyanides_curated_v3.csv" \
    "$REPO/LiON/all_data_extra_x.csv" \
    $(for i in $(seq 0 9); do echo "$REPO/LiON/cv_splits/cv_$i/trained_model_checkpoints"; done)
