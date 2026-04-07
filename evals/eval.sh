#!/bin/bash
CONFIG_FILE="eval_config.yaml"
CHECKLIST="completed_runs.txt"
# ---------------------

echo ">>> Syncing with WandB to fetch updated run list..."
python sync_wandb.py

# Ensure file exists even if sync failed or projects were empty
touch "$CHECKLIST"

# 1. Handle Step Input
if [[ -z "$1" || "$1" == "all" ]]; then
    STEP_PATTERN="*"
else
    STEP_PATTERN="$1"
fi
shift

# 2. Define Models list
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=($(find "$WORK/bayesrl" -maxdepth 1 -mindepth 1 -type d -exec basename {} \;))
fi

echo ">>> Starting Evaluation Pipeline"
echo ">>> Step Pattern: $STEP_PATTERN"

for MODEL in "${MODELS[@]}"; do
    MODEL_BASE_PATH="$WORK/bayesrl/$MODEL"
    
    if [[ ! -d "$MODEL_BASE_PATH" ]]; then
        echo "Skipping: $MODEL (Directory not found)"
        continue
    fi

    # Locate the HF subdirectories
    HF_DIRS=($(find "$MODEL_BASE_PATH" -maxdepth 2 -type d -path "*/global_step_${STEP_PATTERN}/huggingface" | sort -V))

    if [ ${#HF_DIRS[@]} -eq 0 ]; then
        continue
    fi

    for HF_PATH in "${HF_DIRS[@]}"; do
        STEP_NAME=$(basename "$(dirname "$HF_PATH")")
        RUN_NAME="${MODEL}--${STEP_NAME}"

        # CHECK AGAINST CACHED LIST
        if grep -Fxq "$RUN_NAME" "$CHECKLIST"; then
            echo ">>> [SKIP] $RUN_NAME already exists in WandB."
            continue
        fi

        echo "========================================================"
        echo "EVALUATING: $RUN_NAME"
        echo "PATH: $HF_PATH"
        echo "========================================================"

        # Run the evaluation script
        python math_evals.py \
            --config "$CONFIG_FILE" \
            model.path="$HF_PATH" \
            wandb.name="$RUN_NAME"
            
        # If successful, append to local file so we don't need to re-sync 
        # with the cloud for every single iteration within this loop.
        if [ $? -eq 0 ]; then
            echo "$RUN_NAME" >> "$CHECKLIST"
            echo "Successfully finished: $RUN_NAME"
        else
            echo "!!! FAILED: $RUN_NAME (Check logs)"
        fi
    done
done

echo ">>> Evaluation Pipeline Complete."