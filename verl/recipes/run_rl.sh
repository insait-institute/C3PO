#!/bin/bash
set -x

# --- 1. CONFIGURATION / PARAMETERS ---
# Model Mapping Logic
MODEL_NAME=${MODEL_NAME:-"olmo3"}   
MODEL_TYPE=${MODEL_TYPE:-"base"}    
DEFAULT_TOKENIZER_PATH=null
DATA_ROOT=${DATA_ROOT:-"${HOME}/bayesrl/verl/data"}
DATA_NAME=${DATA_NAME:-"dapomath"}

if [ "$MODEL_NAME" == "olmo3" ]; then
    if [ "$MODEL_TYPE" == "base" ]; then
        MODEL_PATH="allenai/Olmo-3-1025-7B"
        DEFAULT_TOKENIZER_PATH="allenai/Olmo-3-7B-Think-DPO"
    else
        MODEL_PATH="allenai/Olmo-3-7B-Think-DPO"
    fi
    TRAIN_DATA=$DATA_ROOT/${MODEL_NAME}-${MODEL_TYPE}-${DATA_NAME}-train.parquet
elif [ "$MODEL_NAME" == "qwm" ]; then
    if [ "$MODEL_TYPE" == "base" ]; then
        MODEL_PATH="Qwen/Qwen2.5-Math-7B"
    else
        MODEL_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
    fi
    TRAIN_DATA=$DATA_ROOT/${MODEL_NAME}-${MODEL_TYPE}-${DATA_NAME}-train.parquet
elif [ "$MODEL_NAME" == "qwm_nmtron" ]; then
    MODEL_PATH="BayesRL/qwm7b_nmtron_ivon"
    TRAIN_DATA="$DATA_ROOT/qwm-instruct-${DATA_NAME}-train.parquet"
elif [ "$MODEL_NAME" == "olmo3_nmtron" ]; then
    MODEL_PATH="BayesRL/olmo3_nmtron_ivon"
    TRAIN_DATA="$DATA_ROOT/olmo3-instruct-${DATA_NAME}-train.parquet"
elif [ "$MODEL_NAME" == "llama_nmtron" ]; then
    MODEL_PATH="BayesRL/llama_nmtron_ivon"
    TRAIN_DATA="$DATA_ROOT/llama-instruct-${DATA_NAME}-train.parquet"
else
    MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-Math-7B"}
    TRAIN_DATA="$DATA_ROOT/qwm-base-${DATA_NAME}-train.parquet"
fi
TOKENIZER_PATH=${TOKENIZER_PATH:-$DEFAULT_TOKENIZER_PATH}
IVON_INIT_METHOD=${IVON_INIT_METHOD:-"scratch"} # scratch or trained

# Basic Training Params
OPTIMIZER=${OPTIMIZER:-"adamw"}   
METHOD=${METHOD:-"grpo"}       

# Hyperparameters based on MODEL_TYPE and OPTIMIZER
if [ "$MODEL_TYPE" == "instruct" ]; then
    if [ "$OPTIMIZER" == "ivon" ]; then
        DEFAULT_LR=1.0
        DEFAULT_WD=1e-6
        DEFAULT_MAX_TOKEN_LEN=25000  # IVON is heavier on memory
    else
        DEFAULT_LR=1e-6
        DEFAULT_WD=1e-1
        DEFAULT_MAX_TOKEN_LEN=30000
    fi
else
    if [ "$OPTIMIZER" == "ivon" ]; then
        DEFAULT_LR=1.0
        DEFAULT_WD=1e-6
        DEFAULT_MAX_TOKEN_LEN=25000
    else
        DEFAULT_LR=1e-6
        DEFAULT_WD=1e-1
        DEFAULT_MAX_TOKEN_LEN=30000
    fi
fi
# Initial parameter assignment
LR=${LR:-$DEFAULT_LR}
WD=${WD:-$DEFAULT_WD}
MAX_TOKEN_LEN=${MAX_TOKEN_LEN:-$DEFAULT_MAX_TOKEN_LEN}
KL_COEF=${KL_COEF:-0}
ENTROPY_COEF=${ENTROPY_COEF:-0}
CLIP_LOW=${CLIP_LOW:-0.2}
CLIP_HIGH=${CLIP_HIGH:-0.2}
KL_COV_RATIO=${KL_COV_RATIO:--1}
PPO_KL_COEF=${PPO_KL_COEF:--1}
CLIP_COV_RATIO=${CLIP_COV_RATIO:--1}
CLIP_COV_LB=${CLIP_COV_LB:--1}
CLIP_COV_UB=${CLIP_COV_UB:--1}
MC_SAMPLES=${MC_SAMPLES:-1}
NUM_EPOCHS=${NUM_EPOCHS:-3}
GROUP_SIZE=${GROUP_SIZE:-8}

if [ "$OPTIMIZER" == "ivon" ]; then
    BETAS=${BETAS:-"[0.9,0.9999]"}
    ESS=${ESS:-1e9}
    ESS_SCHEDULE=${ESS_SCHEDULE:-"constant"}
    MIN_ESS=${MIN_ESS:-$ESS}
    EXPLORATION_M=${EXPLORATION_M:-1}
else
    BETAS=${BETAS:-"[0.9,0.999]"}
fi

# --- 2. METHOD & EXPERIMENT NAMING ---
EXPNAME=${EXPNAME:-"${MODEL_NAME}-${MODEL_TYPE}-${OPTIMIZER}-${DATA_NAME}-LR_${LR}-GS_${GROUP_SIZE}"}

if [ "$OPTIMIZER" == "ivon" ]; then
    EXPNAME="${EXPNAME}-ESS${ESS}-IVONINIT_${IVON_INIT_METHOD}"
fi

# Apply Method-specific overrides and name updates
if [ "$METHOD" == "grpo_cliphigh" ]; then
    CLIP_HIGH=0.5
    EXPNAME="${EXPNAME}-CLIPHIGH${CLIP_HIGH}"
elif [ "$METHOD" == "grpo_klcov" ]; then
    KL_COV_RATIO=0.2
    PPO_KL_COEF=1
    EXPNAME="${EXPNAME}-KLCOV${KL_COV_RATIO}-PPOKL${PPO_KL_COEF}"
elif [ "$METHOD" == "grpo_entloss" ]; then
    ENTROPY_COEF=5e-3
    EXPNAME="${EXPNAME}-ENTCOEF${ENTROPY_COEF}"
elif [ "$METHOD" == "grpo_clipcov" ]; then
    CLIP_LOW=1
    CLIP_HIGH=1
    CLIP_COV_RATIO=0.0002
    CLIP_COV_LB=1.0
    CLIP_COV_UB=5.0
    EXPNAME="${EXPNAME}-CLIPCOV${CLIP_COV_RATIO}-CLIPCOVLB${CLIP_COV_LB}-CLIPCOVUB${CLIP_COV_UB}"
fi

if [ -n "$ESS_SCHEDULE" ] && [ "$ESS_SCHEDULE" != "constant" ]; then
    EXPNAME="${EXPNAME}-SCHED_${ESS_SCHEDULE}"
fi
if [ "$MIN_ESS" != "$ESS" ] && [ "$OPTIMIZER" == "ivon" ]; then
    EXPNAME="${EXPNAME}-MINESS_${MIN_ESS}"
fi
if [ "$MC_SAMPLES" != 1 ]; then
    EXPNAME="${EXPNAME}-MCSAMPLES${MC_SAMPLES}"
fi
if (( "$EXPLORATION_M" > 1 )); then
    BYPASS_MODE=false
    EXPNAME="${EXPNAME}-interleave${EXPLORATION_M}"
else
    BYPASS_MODE=true
fi

# --- 3. DYNAMIC ARGUMENT CONSTRUCTION ---
# Handle KL_Cov/Clip_Cov logic
KL_COV_LINE=""
if [ "$KL_COV_RATIO" != "-1" ] && [ "$PPO_KL_COEF" != "-1" ]; then
    KL_COV_LINE="actor_rollout_ref.actor.kl_ctrl.kl_cov_ratio=$KL_COV_RATIO actor_rollout_ref.actor.policy_loss.ppo_kl_coef=$PPO_KL_COEF"
fi

CLIP_COV_LINE=""
if [ "$CLIP_COV_RATIO" != "-1" ] && [ "$CLIP_COV_LB" != "-1" ] && [ "$CLIP_COV_UB" != "-1" ]; then
    CLIP_COV_LINE="actor_rollout_ref.actor.policy_loss.clip_cov_ratio=$CLIP_COV_RATIO actor_rollout_ref.actor.policy_loss.clip_cov_lb=$CLIP_COV_LB actor_rollout_ref.actor.policy_loss.clip_cov_ub=$CLIP_COV_UB"
fi

# Optimizer args
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

LR_WARMUP_STEPS=34
EVAL_FREQ=28
if [[ "$DATA_NAME" == "dapomath-dc1024" ]]; then
    LR_WARMUP_STEPS=7
    EVAL_FREQ=16
elif [[ "$DATA_NAME" == "skywork_hard" ]]; then
    LR_WARMUP_STEPS=10
    EVAL_FREQ=10
fi

SAVE_FREQ=${SAVE_FREQ:-0.25}

# --- 4. PATHS & ENV ---
nnodes=1
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
project_name=VeRL-RL

SAVE_ROOT=${SAVE_ROOT:-"${WORK}/bayesrl"}
EVAL_DATA=$DATA_ROOT/math_evals.parquet
SAVE_PATH=$SAVE_ROOT/$EXPNAME

# --- 5. EXECUTION ---
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    algorithm.rollout_correction.rollout_is=sequence \
    algorithm.rollout_correction.bypass_mode=$BYPASS_MODE \
    algorithm.rollout_correction.loss_type=ppo_clip \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.train_batch_size=32 \
    data.shuffle=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.tokenizer_path=$TOKENIZER_PATH \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=$CLIP_LOW \
    actor_rollout_ref.actor.clip_ratio_high=$CLIP_HIGH \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=$ENTROPY_COEF \
    actor_rollout_ref.actor.kl_loss_coef=$KL_COEF \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.actor.optim.betas=$BETAS \
    actor_rollout_ref.actor.optim.weight_decay=$WD \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS \
    actor_rollout_ref.actor.optim.lr_scheduler_type=constant \
    $OPT_ARGS \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.calculate_entropy=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$MAX_TOKEN_LEN \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.M=$EXPLORATION_M \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=50 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    critic.model.tokenizer_path=$TOKENIZER_PATH \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$EXPNAME \
    trainer.n_gpus_per_node=$nproc_per_node \
    trainer.logger='["console","wandb"]' \
    trainer.critic_warmup=0 \
    trainer.total_epochs=$NUM_EPOCHS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$EVAL_FREQ \
    trainer.log_completions_freq=0.025 \
    trainer.val_before_train=False \
    trainer.nnodes=$nnodes \
    trainer.n_gpus_per_node=$nproc_per_node \
    trainer.validation_data_dir=$SAVE_PATH/eval_gens \
    $KL_COV_LINE \
    $CLIP_COV_LINE