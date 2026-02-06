#!/usr/bin/env bash
set -euo pipefail

TUNE_DIR=${TUNE_DIR:-results_tuning}
OUTDIR=${OUTDIR:-results}

EVAL_EPISODES=${EVAL_EPISODES:-10000}
DQ_EPISODES=${DQ_EPISODES:-200000}

A2C_STEPS=${A2C_STEPS:-200000}
A2C_ROLLOUT=${A2C_ROLLOUT:-256}
A2C_DEVICE=${A2C_DEVICE:-cpu}

pick_best_args() {
  local algo=$1
  local reward=$2

  uv run python - <<'PY' "$algo" "$reward" "$TUNE_DIR"
import json
import sys
from pathlib import Path

algo = sys.argv[1]
reward = sys.argv[2]
tune_dir = Path(sys.argv[3])

best_path = None
best_score = None
best_data = None

for path in tune_dir.glob(f"tune_{algo}_{reward}_*/best_config.json"):
    with path.open() as handle:
        data = json.load(handle)
    score = data.get("mean_eval_return", float("-inf"))
    if best_score is None or score > best_score:
        best_score = score
        best_path = path
        best_data = data

if best_path is None or best_data is None:
    raise SystemExit(f"No tuning results for {algo} {reward} in {tune_dir}")

if algo == "doubleq":
    print(
        "--alpha",
        best_data["alpha"],
        "--gamma",
        best_data["gamma"],
        "--eps_end",
        best_data["eps_end"],
        "--eps_decay_episodes",
        int(best_data["eps_decay_episodes"]),
    )
else:
    print(
        "--lr",
        best_data["lr"],
        "--gamma",
        best_data["gamma"],
        "--gae_lambda",
        best_data["gae_lambda"],
        "--entropy_coef",
        best_data["entropy_coef"],
        "--hidden1",
        int(best_data["hidden1"]),
        "--hidden2",
        int(best_data["hidden2"]),
    )
PY
}

for seed in 0 1 2; do
  for reward in r0 r1 r2; do
    EXTRA_ARGS=()
    if [ "$reward" = "r1" ]; then
      EXTRA_ARGS+=(--step_penalty 0.01)
    elif [ "$reward" = "r2" ]; then
      EXTRA_ARGS+=(--bust_penalty 0.5)
    fi

    echo "=== DoubleQ | $reward | seed=$seed ==="
    read -r -a DQ_ARGS <<< "$(pick_best_args doubleq "$reward")"
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
      uv run python -m src.run \
        --algo doubleq \
        --reward "$reward" \
        --seed "$seed" \
        --train_episodes "$DQ_EPISODES" \
        --eval_episodes "$EVAL_EPISODES" \
        --outdir "$OUTDIR" \
        "${DQ_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
    else
      uv run python -m src.run \
        --algo doubleq \
        --reward "$reward" \
        --seed "$seed" \
        --train_episodes "$DQ_EPISODES" \
        --eval_episodes "$EVAL_EPISODES" \
        --outdir "$OUTDIR" \
        "${DQ_ARGS[@]}"
    fi

    echo "=== A2C | $reward | seed=$seed ==="
    read -r -a A2C_ARGS <<< "$(pick_best_args a2c "$reward")"
    if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
      uv run python -m src.run \
        --algo a2c \
        --reward "$reward" \
        --seed "$seed" \
        --train_steps "$A2C_STEPS" \
        --rollout_steps "$A2C_ROLLOUT" \
        --eval_episodes "$EVAL_EPISODES" \
        --device "$A2C_DEVICE" \
        --outdir "$OUTDIR" \
        "${A2C_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
    else
      uv run python -m src.run \
        --algo a2c \
        --reward "$reward" \
        --seed "$seed" \
        --train_steps "$A2C_STEPS" \
        --rollout_steps "$A2C_ROLLOUT" \
        --eval_episodes "$EVAL_EPISODES" \
        --device "$A2C_DEVICE" \
        --outdir "$OUTDIR" \
        "${A2C_ARGS[@]}"
    fi
  done
done
