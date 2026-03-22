#!/bin/bash
set -x

LR=${LR:-5}
WD=${WD:-1e-6}
KL_COEF=${KL_COEF:-0}
ENTROPY_COEF=${ENTROPY_COEF:-0}
CLIP_LOW=${CLIP_LOW:-0.2}
CLIP_HIGH=${CLIP_HIGH:-0.2}
KL_COV_RATIO=${KL_COV_RATIO:--1}
PPO_KL_COEF=${PPO_KL_COEF:--1}
CLIP_COV_RATIO=${CLIP_COV_RATIO:--1}
CLIP_COV_LB=${CLIP_COV_LB:--1}
CLIP_COV_UB=${CLIP_COV_UB:--1}


if [ "$KL_COV_RATIO" != -1 ] && [ "$PPO_KL_COEF" != -1 ]; then
    KL_COV_LINE="actor_rollout_ref.actor.kl_ctrl.kl_cov_ratio=$KL_COV_RATIO actor_rollout_ref.actor.policy_loss.ppo_kl_coef=$PPO_KL_COEF"
else
    KL_COV_LINE=""
fi

if [ "$CLIP_COV_RATIO" != -1 ] && [ "$CLIP_COV_LB" != -1 ] && [ "$CLIP_COV_UB" != -1 ]; then
    CLIP_COV_LINE="actor_rollout_ref.actor.policy_loss.clip_cov_ratio=$CLIP_COV_RATIO actor_rollout_ref.actor.policy_loss.clip_cov_lb=$CLIP_COV_LB actor_rollout_ref.actor.policy_loss.clip_cov_ub=$CLIP_COV_UB"
else
    CLIP_COV_LINE=""
fi

ESS=${ESS:-1}
ESS_SCHEDULE=${ESS_SCHEDULE:-constant}


nnodes=1
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
project_name=VeRL-RL
experiment_name=${EXPNAME:-qwm7b-scratch-dapomath-ivon}

DATA_ROOT=${DATA_ROOT:-"${HOME}/bayesrl/verl/data"}
SAVE_ROOT=${SAVE_ROOT:-"${WORK}/bayesrl"}
TRAIN_DATA=$DATA_ROOT/qwm-dapomath-base-train.parquet
EVAL_DATA=null
MODEL_PATH=Qwen/Qwen2.5-Math-7B
SAVE_PATH=$SAVE_ROOT/$experiment_name

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.train_batch_size=32 \
    data.shuffle=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=50000 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_HIGH \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEF \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.calculate_entropy=$([[ "$ESS_SCHEDULE" =~ "adaptive" ]] && echo "True" || echo "False") \
    actor_rollout_ref.actor.optim.optimizer=IVON \
    actor_rollout_ref.actor.optim.optimizer_impl=ivon \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.betas=[0.9,0.9999] \
    actor_rollout_ref.actor.optim.weight_decay=$WD \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.optim.override_optimizer_config="{ess:1e8,hess_init:0.001,clip_radius:1e-3,rescale_lr:True,sync:false,initial_ess_scale:$ESS,ess_schedule:$ESS_SCHEDULE,min_ess_scale:0}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=50000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
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
    $CLIP_COV_LINE 
    # +actor_rollout_ref.actor.optim.optimizer_load_path=$MODEL_PATH \