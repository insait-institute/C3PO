#!/bin/bash
CONFIG_FILE="eval_config.yaml"
CHECKLIST="completed_runs.txt"
# ---------------------

# --- SHARDING LOGIC ---
# Default to Rank 0 of 1 (no sharding) if variables aren't set
NODE_RANK=${NODE_RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}

echo ">>> Sharding Config: Rank $NODE_RANK / Total Shards $WORLD_SIZE"
# ---------------------

echo ">>> Syncing with WandB..."
python sync_wandb.py
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

# 3. Collect ALL valid HF paths into an array first
echo ">>> Scanning for evaluation targets..."
ALL_HF_PATHS=()
for MODEL in "${MODELS[@]}"; do   
    MODEL_BASE_PATH="$WORK/bayesrl/$MODEL"
    if [[ -d "$MODEL_BASE_PATH" ]]; then
        # Find paths and add to master list
        while read -r path; do
            ALL_HF_PATHS+=("$path")
        done < <(find "$MODEL_BASE_PATH" -maxdepth 2 -type d -path "*/global_step_${STEP_PATTERN}/huggingface" | sort -V)
    fi
done

TOTAL_RUNS=${#ALL_HF_PATHS[@]}
echo ">>> Found $TOTAL_RUNS total candidate runs."

# 4. Iterate with Modulo Sharding
COUNTER=0
PROCESSED_COUNT=0

for HF_PATH in "${ALL_HF_PATHS[@]}"; do
    # Check if this run belongs to this shard
    if [ $((COUNTER % WORLD_SIZE)) -ne $NODE_RANK ]; then
        ((COUNTER++))
        continue
    fi
    ((COUNTER++))

    STEP_NAME=$(basename "$(dirname "$HF_PATH")")
    # Extract model name from the path for the RUN_NAME
    MODEL_NAME=$(basename "$(dirname "$(dirname "$HF_PATH")")")
    RUN_NAME="${MODEL_NAME}--${STEP_NAME}"

    # CHECK AGAINST CACHED LIST
    if grep -Fxq "$RUN_NAME" "$CHECKLIST"; then
        echo ">>> [SKIP] $RUN_NAME already exists on WandB."
        continue
    fi

    echo "========================================================"
    echo "RANK $NODE_RANK | EVALUATING: $RUN_NAME"
    echo "PATH: $HF_PATH"
    echo "========================================================"

    python math_evals.py \
        --config "$CONFIG_FILE" \
        model.path="$HF_PATH" \
        wandb.name="$RUN_NAME"
        
    if [ $? -eq 0 ]; then
        echo "$RUN_NAME" >> "$CHECKLIST"
        ((PROCESSED_COUNT++))
    else
        echo "!!! FAILED: $RUN_NAME"
    fi
done

echo ">>> Rank $NODE_RANK complete. Processed $PROCESSED_COUNT runs."