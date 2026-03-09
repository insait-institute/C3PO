#!/bin/bash
set -x

LR=${LR:-2}
ESS=${ESS:-1}
ESS_SCHEDULE=${ESS_SCHEDULE:-constant}
nnodes=1
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
project_name=VeRL-RL
experiment_name=${EXPNAME:-ivon-olmo2-1b-rl-dapomath}

DATA_ROOT=${DATA_ROOT:-"${HOME}/bayesrl/verl/data"}
SAVE_ROOT=${SAVE_ROOT:-"${WORK}/bayesrl"}
TRAIN_DATA=$DATA_ROOT/dapomath-train.parquet
EVAL_DATA=$DATA_ROOT/gsm8k-test.parquet
MODEL_PATH=BayesRL/ivon-1b-sft
SAVE_PATH=$SAVE_ROOT/$experiment_name

PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0 \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_prompt_length=1536 \
    data.max_response_length=3072 \
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
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.calculate_entropy=$([[ "$ESS_SCHEDULE" =~ "adaptive" ]] && echo "True" || echo "False") \
    actor_rollout_ref.actor.optim.optimizer=IVON \
    actor_rollout_ref.actor.optim.optimizer_impl=ivon \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.betas=[0.9,0.99999] \
    actor_rollout_ref.actor.optim.weight_decay=1e-4 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.optim.override_optimizer_config="{ess:1e8,hess_init:0.001,clip_radius:1e-3,rescale_lr:True,sync:false,initial_ess_scale:$ESS,ess_schedule:$ESS_SCHEDULE,min_ess_scale:0}" \
    +actor_rollout_ref.actor.optim.optimizer_load_path=$MODEL_PATH \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=50000 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
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
    trainer.total_epochs=1 \
    trainer.save_freq=0 \
    trainer.test_freq=0 \
    trainer.val_before_train=False\
    trainer.nnodes=$nnodes \
    trainer.n_gpus_per_node=$nproc_per_node


#     algorithm.rollout_correction.rollout_is=sequence \
#     algorithm.rollout_correction.rollout_is_threshold=2.0 \
#     algorithm.rollout_correction.bypass_mode=True \
#     actor_rollout_ref.rollout.calculate_log_probs=True \
