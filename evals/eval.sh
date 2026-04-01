#!/bin/bash
export HOME=$VV_WORKDIR
source $HOME/init.sh
micromamba activate verlivon

# --- CONFIGURATION ---
CONFIG_FILE="eval_config.yaml"
# ---------------------

# 1. Handle Step Input (all or a specific number)
if [[ -z "$1" || "$1" == "all" ]]; then
    STEP_PATTERN="*"
else
    STEP_PATTERN="$1"
fi
shift

# 2. Define Models list (remaining arguments or all in bayesrl)
if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=($(find "$WORK/bayesrl" -maxdepth 1 -mindepth 1 -type d -exec basename {} \;))
fi

echo ">>> Starting Evaluation Pipeline"
echo ">>> Step Pattern: $STEP_PATTERN"
echo ">>> Models: ${MODELS[*]}"

for MODEL in "${MODELS[@]}"; do
    MODEL_BASE_PATH="$WORK/bayesrl/$MODEL"
    
    if [[ ! -d "$MODEL_BASE_PATH" ]]; then
        echo "Skipping: $MODEL (Directory not found)"
        continue
    fi

    # Find the huggingface subdirectories inside the global_step folders
    # We look specifically for the 'huggingface' folder created by your merger script
    HF_DIRS=($(find "$MODEL_BASE_PATH" -maxdepth 2 -type d -path "*/global_step_${STEP_PATTERN}/huggingface" | sort -V))

    if [ ${#HF_DIRS[@]} -eq 0 ]; then
        echo "No merged HuggingFace models found for $MODEL at step $STEP_PATTERN"
        continue
    fi

    for HF_PATH in "${HF_DIRS[@]}"; do
        # Extract the step name for logging/wandb (e.g., global_step_580)
        STEP_NAME=$(basename "$(dirname "$HF_PATH")")
        
        echo "========================================================"
        echo "EVALUATING: $MODEL | $STEP_NAME"
        echo "PATH: $HF_PATH"
        echo "========================================================"

        # Run the evaluation script
        # We override model.path and wandb.name via CLI arguments
        python math_evals.py \
            --config eval_config.yaml \
            model.path="$HF_PATH" \
            wandb.name="${MODEL}_${STEP_NAME}"
            
        echo "Finished eval for $MODEL at $STEP_NAME"
    done
done

echo ">>> Evaluation Pipeline Complete."