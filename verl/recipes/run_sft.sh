#!/bin/bash
set -x
nnodes=1
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
if [ "$nproc_per_node" -eq 0 ]; then
    nproc_per_node=1
fi
master_port=${MASTER_PORT:-29500}
project_name=VeRL-SFT

# --- 1. CONFIGURATION / PARAMETERS ---
MODEL_NAME=${MODEL_NAME:-"olmo3"}   
MODEL_SIZE=${MODEL_SIZE:-"7B"}    
DATA_NAME=${DATA_NAME:-"nmt"}
OPTIMIZER=${OPTIMIZER:-"ivon"}

DATA_ROOT=${DATA_ROOT:-"${HOME}/bayesrl/verl/data"}
SAVE_ROOT=${SAVE_ROOT:-"${WORK}/bayesrl"}

# Dataset mapping
if [ "$DATA_NAME" == "tulu" ]; then
    TRAIN_DATA=$DATA_ROOT/tulu-3-sft-olmo-2-mixture-0225.parquet
elif [ "$DATA_NAME" == "nmt" ]; then
    TRAIN_DATA=$DATA_ROOT/nemotron_ptds_${MODEL_NAME}.parquet
else
    TRAIN_DATA=${TRAIN_DATA:-$DATA_ROOT/${DATA_NAME}}
fi
EVAL_DATA=null

# Model mapping & Optimizer Defaults
if [ "$MODEL_NAME" == "olmo3" ]; then
    MODEL_PATH="allenai/Olmo-3-1025-${MODEL_SIZE}"
    TOKENIZER_PATH="allenai/Olmo-3-${MODEL_SIZE}-Think-DPO"
elif [ "$MODEL_NAME" == "qwm" ]; then
    MODEL_PATH="Qwen/Qwen2.5-Math-${MODEL_SIZE}"
    TOKENIZER_PATH="Qwen/Qwen2.5-Math-${MODEL_SIZE}-Instruct"
elif [ "$MODEL_NAME" == "llama" ]; then
    MODEL_PATH="meta-llama/Llama-3.1-${MODEL_SIZE}"
    TOKENIZER_PATH="meta-llama/Llama-3.1-${MODEL_SIZE}-Instruct"
else
    MODEL_PATH=$MODEL_NAME
    TOKENIZER_PATH=$MODEL_NAME
fi

if [ "$OPTIMIZER" == "ivon" ]; then
    DEFAULT_MBS=2
    DEFAULT_LR=50
    DEFAULT_WD=1e-8
    DEFAULT_ESS=1e10
    DEFAULT_BETAS="[0.9,0.9999]"
else
    DEFAULT_MBS=4
    DEFAULT_LR=1e-5
    DEFAULT_WD=0.1
    DEFAULT_BETAS="[0.9,0.999]"
fi

# Initial parameter assignment
LR=${LR:-$DEFAULT_LR}
WD=${WD:-$DEFAULT_WD}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-$DEFAULT_MBS}
BETAS=${BETAS:-$DEFAULT_BETAS}
NUM_EPOCHS=${NUM_EPOCHS:-2}
WARMUP_RATIO=${WARMUP_RATIO:-0.1}

if [ "$OPTIMIZER" == "ivon" ]; then
    ESS=${ESS:-$DEFAULT_ESS}
fi

# --- 2. EXPERIMENT NAMING ---
EXPNAME=${EXPNAME:-"${OPTIMIZER}-${MODEL_NAME}-${MODEL_SIZE}-sft-${DATA_NAME}-LR_${LR}"}

if [ "$OPTIMIZER" == "ivon" ]; then
    EXPNAME="${EXPNAME}-ESS${ESS}"
fi

# --- 3. DYNAMIC ARGUMENT CONSTRUCTION ---
# Optimizer args
if [ "$OPTIMIZER" == "ivon" ]; then
    OPT_ARGS="
        optim.optimizer=IVON \
        optim.optimizer_impl=ivon \
        optim.ivon_config.ess=$ESS \
        optim.ivon_config.hess_init=0.001 \
        optim.ivon_config.clip_radius=1e-3 \
        optim.ivon_config.rescale_lr=True \
        optim.ivon_config.sync=false \
    "
else
    OPT_ARGS="
        optim.optimizer=AdamW \
        optim.optimizer_impl=torch.optim
    "
fi

SAVE_PATH=$SAVE_ROOT/$EXPNAME

# --- 5. EXECUTION ---
torchrun --nnodes=$nnodes \
     --nproc_per_node=$nproc_per_node \
     --master_port=$master_port \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=4096 \
    data.train_batch_size=128 \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.tokenizer_path=$TOKENIZER_PATH \
    data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp2 \
    model.use_liger=true \
    model.fsdp_config.model_dtype=bfloat16 \
    model.fsdp_config.cpu_offload=false \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$EXPNAME \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.save_freq=0.25 \
    trainer.test_freq=0.1 \
    trainer.val_before_train=true \
    trainer.nnodes=$nnodes \
    trainer.n_gpus_per_node=$nproc_per_node \
    model.enable_gradient_checkpointing=false \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true \
    $OPT_ARGS \
    optim.lr=$LR \
    optim.betas=$BETAS \
    optim.weight_decay=$WD \
    optim.lr_warmup_steps_ratio=$WARMUP_RATIO \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    optim.min_lr_ratio=0.1
