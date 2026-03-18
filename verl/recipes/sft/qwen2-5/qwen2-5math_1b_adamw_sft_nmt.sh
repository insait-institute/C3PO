#!/bin/bash
set -x

LR=${LR:-1e-5}
WD=${WD:-0.1}

nnodes=1
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
master_port=${MASTER_PORT:-29500}
project_name=VeRL-SFT
experiment_name=${EXPNAME:-adamw-qwen2-5math-1b-sft-nmt}

DATA_ROOT=${DATA_ROOT:-"${HOME}/bayesrl/verl/data"}
SAVE_ROOT=${SAVE_ROOT:-"${WORK}/bayesrl"}
TRAIN_DATA=$DATA_ROOT/nemotron_ptds.parquet
EVAL_DATA=null
MODEL_PATH=Qwen/Qwen2.5-Math-1.5B
TOKENIZER_PATH=Qwen/Qwen2.5-Math-1.5B-Instruct
SAVE_PATH=$SAVE_ROOT/$experiment_name

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
    data.micro_batch_size_per_gpu=8 \
    model.partial_pretrain=$MODEL_PATH \
    model.strategy=fsdp2 \
    model.use_liger=true \
    model.fsdp_config.model_dtype=bfloat16 \
    model.fsdp_config.cpu_offload=false \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=1 \
    trainer.save_freq=0.1 \
    trainer.test_freq=0 \
    trainer.nnodes=$nnodes \
    trainer.n_gpus_per_node=$nproc_per_node \
    model.enable_gradient_checkpointing=false \
    ulysses_sequence_parallel_size=1 \
    use_remove_padding=true \
    optim.optimizer=AdamW \
    optim.optimizer_impl=torch.optim \
    optim.lr=$LR \
    optim.betas=[0.9,0.999] \
    optim.weight_decay=$WD \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    optim.min_lr_ratio=0.1