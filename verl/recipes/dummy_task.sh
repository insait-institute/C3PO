#!/bin/bash
set -x


nnodes=1
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
project_name=VeRL-RL
experiment_name=${EXPNAME:-adamw-olmo2-1b-rl}

DATA_ROOT=${DATA_ROOT:-"${HOME}/bayesrl/verl/data"}
SAVE_ROOT=${SAVE_ROOT:-"${WORK}/bayesrl"}
TRAIN_DATA=$DATA_ROOT/dummy.parquet
MODEL_PATH=Qwen/Qwen2.5-3B
SAVE_PATH=$SAVE_ROOT/$experiment_name
OPTIMIZER=${OPTIMIZER:-"adamw"}   

if [ "$OPTIMIZER" == "ivon" ]; then
    BETAS=${BETAS:-"[0.9,0.9999]"}
    ESS=${ESS:-1e9}
    ESS_SCHEDULE=${ESS_SCHEDULE:-"constant"}
    MIN_ESS=${MIN_ESS:-$ESS}
    MC_SAMPLES=${MC_SAMPLES:-1}
    LR=${LR:-1}
    WD=${WD:-1e-6}
else
    BETAS=${BETAS:-"[0.9,0.999]"}
    LR=${LR:-1e-6}
    WD=${WD:-1e-1}
fi

OPT_ARGS=""
if [ "$OPTIMIZER" == "ivon" ]; then
    OPT_ARGS="
        actor_rollout_ref.actor.optim.optimizer=IVON \
        actor_rollout_ref.actor.optim.optimizer_impl=ivon \
        actor_rollout_ref.actor.optim.ivon_config.ess=$ESS \
        actor_rollout_ref.actor.optim.ivon_config.hess_init=0.001 \
        actor_rollout_ref.actor.optim.ivon_config.clip_radius=1e-3 \
        actor_rollout_ref.actor.optim.ivon_config.rescale_lr=True \
        actor_rollout_ref.actor.optim.ivon_config.sync=false \
        actor_rollout_ref.actor.optim.ivon_config.ess_schedule=$ESS_SCHEDULE \
        actor_rollout_ref.actor.optim.ivon_config.min_ess=$MIN_ESS \
        actor_rollout_ref.actor.optim.ivon_config.mc_samples=$MC_SAMPLES
    "
    if [ "$IVON_INIT_METHOD" == "trained" ]; then
        OPT_ARGS="${OPT_ARGS} \
            +actor_rollout_ref.actor.optim.optimizer_load_path=$MODEL_PATH
        "
    fi
else
    OPT_ARGS="actor_rollout_ref.actor.optim.optimizer=AdamW"
fi

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0 \
    data.train_files=$TRAIN_DATA \
    data.val_files=null \
    data.max_prompt_length=256 \
    data.max_response_length=512 \
    data.train_batch_size=64 \
    data.shuffle=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10000 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.optim.betas=$BETAS \
    actor_rollout_ref.actor.optim.weight_decay=$WD \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    $OPT_ARGS \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=16 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$nproc_per_node \
    trainer.logger='["console","wandb"]' \
    trainer.critic_warmup=0 \
    trainer.total_epochs=5 \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    trainer.log_completions_freq=0.2 \
    trainer.val_before_train=False \
    trainer.nnodes=$nnodes \
    trainer.n_gpus_per_node=$nproc_per_node \
    $KL_COV_LINE \
    $CLIP_COV_LINE \