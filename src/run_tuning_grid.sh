#!/usr/bin/env bash
set -euo pipefail

OUTDIR=${OUTDIR:-results}
EVAL_EPISODES=${EVAL_EPISODES:-20000}

DQ_EPISODES=${DQ_EPISODES:-150000}

A2C_STEPS=${A2C_STEPS:-200000}
A2C_ROLLOUT=${A2C_ROLLOUT:-256}
A2C_DEVICE=${A2C_DEVICE:-cpu}

for reward in r0 r1 r2; do
  EXTRA_ARGS=()
  if [ "$reward" = "r1" ]; then
    EXTRA_ARGS+=(--step_penalty 0.01)
  elif [ "$reward" = "r2" ]; then
    EXTRA_ARGS+=(--bust_penalty 0.5)
  fi

  echo "=== tune DoubleQ | $reward ==="
  if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    uv run python -m src.tune \
      --algo doubleq \
      --reward "$reward" \
      --episodes "$DQ_EPISODES" \
      --eval_episodes "$EVAL_EPISODES" \
      --outdir "$OUTDIR" \
      "${EXTRA_ARGS[@]}"
  else
    uv run python -m src.tune \
      --algo doubleq \
      --reward "$reward" \
      --episodes "$DQ_EPISODES" \
      --eval_episodes "$EVAL_EPISODES" \
      --outdir "$OUTDIR"
  fi

  echo "=== tune A2C | $reward ==="
  if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    uv run python -m src.tune \
      --algo a2c \
      --reward "$reward" \
      --steps "$A2C_STEPS" \
      --rollout_steps "$A2C_ROLLOUT" \
      --eval_episodes "$EVAL_EPISODES" \
      --device "$A2C_DEVICE" \
      --outdir "$OUTDIR" \
      "${EXTRA_ARGS[@]}"
  else
    uv run python -m src.tune \
      --algo a2c \
      --reward "$reward" \
      --steps "$A2C_STEPS" \
      --rollout_steps "$A2C_ROLLOUT" \
      --eval_episodes "$EVAL_EPISODES" \
      --device "$A2C_DEVICE" \
      --outdir "$OUTDIR"
  fi
done
