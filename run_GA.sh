#!/usr/bin/env bash
#
# Run the Ugi/LiON GA. Defaults are the best-practice hyperparameters from
# J. Chem. Phys. 159, 091501 (2023), with a larger population since scoring
# cost here is near-independent of population size.
#
#   bash run_GA.sh A                          # production run, label A
#   bash run_GA.sh SMOKE --pop_size 8 --n_generations 3 --conv_gen 3
#   nohup bash run_GA.sh A > /dev/null 2>&1 & # detached on a remote
#
# Any extra arguments are forwarded to GA_main.py and override the defaults.
# Progress goes to results/run_<LABEL>/run.log — follow it with:
#   tail -f results/run_<LABEL>/run.log
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_LABEL="${1:?usage: bash run_GA.sh <RUN_LABEL> [extra GA_main.py args]}"
shift
OUT="$REPO/results/run_$RUN_LABEL"

uv sync --quiet
source "$REPO/.venv/bin/activate"
mkdir -p "$OUT"

cd "$REPO/GA_code"

# -u keeps stdout unbuffered so the log stays current while the run is detached
python -u GA_main.py \
    lion "$RUN_LABEL" \
    --output_dir "$OUT" \
    --maximize \
    --pop_size 96 \
    --selection_method tournament_3 \
    --mutation_rate 0.4 \
    --elitism_perc 0.5 \
    --n_generations 500 \
    --spear_thresh 0.8 \
    --conv_gen 50 \
    --n_jobs 10 \
    --aldehydes   "$REPO/aldehydes_curated_v3.csv" \
    --acids       "$REPO/acids_curated_v3.csv" \
    --amines      "$REPO/amines_curated_v3.csv" \
    --isocyanides "$REPO/isocyanides_curated_v3.csv" \
    "$@" \
    "$REPO/LiON/all_data_extra_x.csv" \
    $(for i in $(seq 0 9); do echo "$REPO/LiON/cv_splits/cv_$i/trained_model_checkpoints"; done) \
    2>&1 | tee "$OUT/run.log"
