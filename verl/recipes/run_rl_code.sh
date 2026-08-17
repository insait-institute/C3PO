#!/bin/bash
set -x

# =============================================================================
# run_rl_code.sh -- RL on code generation (mirror of run_rl.sh)
# =============================================================================
# Same model-selection / optimizer (AdamW or IVON) / rollout-correction
# machinery as recipes/run_rl.sh, but trained on code generation instead of
# math:
#   * train data : OctoReasoner/code-r1-12k  (HF repo id, data_source=code_r1)
#   * eval  data : OctoReasoner/lcb_verl      (HF repo id, data_source=livecodebench)
#   * reward     : the DAPO reward manager scoring generations against stdin/stdout
#                  test cases through a SandboxFusion server (reward_model.sandbox_fusion.*)
#
# Any data_source containing "code" (code_r1, liveCODEbench) routes to the
# sandbox checker (see verl/utils/reward_score/__init__.py).
#
# The sandbox itself is vendored in this repo under ../../SandboxFusion. Launch a
# server first (defaults to http://localhost:8080/run_code), e.g. on an HPC node:
#   cd SandboxFusion && make pull-apptainer-images && make start-apptainer-container
# then point SANDBOX_URL at it.
# =============================================================================

# --- 1. CONFIGURATION / PARAMETERS ---
# Model Mapping Logic (identical to run_rl.sh)
MODEL_NAME=${MODEL_NAME:-"olmo3"}
MODEL_TYPE=${MODEL_TYPE:-"base"}
DEFAULT_TOKENIZER_PATH=null

# DEFAULT_MAX_MODEL_LEN tracks each base model's native max_position_embeddings
# (vLLM refuses to serve a max_model_len above it). Qwen2.5-Math-7B (and the
# qwm_nmtron IVON-SFT checkpoint derived from it) only supports 4096; the other
# bases are long-context (>=32k) so 8192 is a safe rollout window for code.
if [ "$MODEL_NAME" == "olmo3" ]; then
    if [ "$MODEL_TYPE" == "base" ]; then
        MODEL_PATH="allenai/Olmo-3-1025-7B"
        DEFAULT_TOKENIZER_PATH="allenai/Olmo-3-7B-Think-DPO"
    else
        MODEL_PATH="allenai/Olmo-3-7B-Think-DPO"
    fi
    DEFAULT_MAX_MODEL_LEN=8192
elif [ "$MODEL_NAME" == "qwm" ]; then
    if [ "$MODEL_TYPE" == "base" ]; then
        MODEL_PATH="Qwen/Qwen2.5-Math-7B"
    else
        MODEL_PATH="Qwen/Qwen2.5-Math-7B-Instruct"
    fi
    DEFAULT_MAX_MODEL_LEN=4096
elif [ "$MODEL_NAME" == "qwm_nmtron" ]; then
    MODEL_PATH="BayesRL/Qwen2.5Math-IVON-SFT-7B"
    DEFAULT_MAX_MODEL_LEN=4096
elif [ "$MODEL_NAME" == "olmo3_nmtron" ]; then
    MODEL_PATH="BayesRL/Olmo3-IVON-SFT-7B"
    DEFAULT_MAX_MODEL_LEN=8192
elif [ "$MODEL_NAME" == "llama_nmtron" ]; then
    MODEL_PATH="BayesRL/Llama3.1-IVON-SFT-8B"
    DEFAULT_MAX_MODEL_LEN=8192
else
    MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-Coder-7B"}
    DEFAULT_MAX_MODEL_LEN=8192
fi
TOKENIZER_PATH=${TOKENIZER_PATH:-$DEFAULT_TOKENIZER_PATH}
IVON_INIT_METHOD=${IVON_INIT_METHOD:-"scratch"} # scratch or trained

# Data (Hugging Face Hub repo ids, loaded directly via data.{train,val}_files)
DATA_NAME=${DATA_NAME:-"coder1"}
TRAIN_DATA=${TRAIN_DATA:-"OctoReasoner/code-r1-12k"}
EVAL_DATA=${EVAL_DATA:-"OctoReasoner/lcb_verl"}

# Sequence lengths: code prompts/solutions are longer than math.
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-2048}
# vLLM rollout context; see DEFAULT_MAX_MODEL_LEN above for the per-model cap.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$DEFAULT_MAX_MODEL_LEN}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-$((MAX_MODEL_LEN * 2))}

# Basic Training Params
OPTIMIZER=${OPTIMIZER:-"adamw"}
METHOD=${METHOD:-"grpo"}

# Hyperparameters based on OPTIMIZER (code uses a single set; mirrors run_rl.sh)
if [ "$OPTIMIZER" == "ivon" ]; then
    DEFAULT_LR=1.0
    DEFAULT_WD=1e-6
    DEFAULT_MAX_TOKEN_LEN=25000  # IVON is heavier on memory
else
    DEFAULT_LR=1e-6
    DEFAULT_WD=1e-1
    DEFAULT_MAX_TOKEN_LEN=30000
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
M3PO_M=${M3PO_M:-1}
DECOUPLED_MC_SAMPLES=${DECOUPLED_MC_SAMPLES:-1}
NUM_EPOCHS=${NUM_EPOCHS:-1}
GROUP_SIZE=${GROUP_SIZE:-8}
C3PO_N=${C3PO_N:-1}
TEMPERATURE=${TEMPERATURE:-1.0}

# Adaptive-temperature entropy-collapse guard (AdamW baseline).
# When ADAPTIVE_TEMP=true, the rollout temperature is raised from TEMPERATURE to
# TEMP_HIGH once the policy entropy H_t drops below LOW_ENT_RATIO * H_0, where H_0
# is the first measured entropy. See docs/adaptive_temperature_baseline.md.
ADAPTIVE_TEMP=${ADAPTIVE_TEMP:-false}
TEMP_HIGH=${TEMP_HIGH:-1.0}
LOW_ENT_RATIO=${LOW_ENT_RATIO:-0.5}

if [ "$OPTIMIZER" == "ivon" ]; then
    BETAS=${BETAS:-"[0.9,0.9999]"}
    ESS=${ESS:-1e9}
    ESS_SCHEDULE=${ESS_SCHEDULE:-"constant"}
    MIN_ESS=${MIN_ESS:-$ESS}
else
    BETAS=${BETAS:-"[0.9,0.999]"}
fi

# --- Sandbox / reward params (defaults ported from multirl's run_coder1.sh) ---
SANDBOX_URL=${SANDBOX_URL:-"http://localhost:8080/run_code"}
# PER-WORKER ceiling on concurrent sandbox requests. The reward runs through the
# experimental reward_loop: reward_model.num_workers (=8) RewardLoopWorker actors,
# each its own process with its own semaphore, so server-facing concurrency is
# num_workers x this value. multirl's bench_concurrency found the server's
# throughput peaks at total ceiling ~= its slot count (48) and DEGRADES above it
# (CPU-throughput-bound), so 8 x 6 = 48 matches the 48-slot server in
# bayesrl_code.sh (this is exactly multirl's run_coder1.sh value).
SANDBOX_MAX_CONCURRENT=${SANDBOX_MAX_CONCURRENT:-6}
SANDBOX_MEMORY_LIMIT_MB=${SANDBOX_MEMORY_LIMIT_MB:-1024}
# Binary verdict (1.0 iff every test case passes); short-circuits on first fail.
SANDBOX_CONTINUOUS=${SANDBOX_CONTINUOUS:-False}
# Per-case compile/run timeout. The historical 10s is tuned for C++ reference
# limits and systematically times out correct-but-slower Python on large inputs
# (~10% of executions), injecting correctness-uncorrelated noise; 20s de-noises.
SANDBOX_TIMEOUT=${SANDBOX_TIMEOUT:-10}
# Ride out a sandbox server restart / saturation (jittered, capped backoff) by
# retrying read timeouts / 503-504 rather than scoring the case as failed.
SANDBOX_RETRY_ON_TIMEOUT=${SANDBOX_RETRY_ON_TIMEOUT:-False}

# Cap glibc malloc arenas: the reward workers fan the sandbox check across a
# thread pool; with the default unbounded arena count the memory freed after
# each scoring batch is retained per-arena and never returned to the OS, so RSS
# ratchets up until the node OOMs. ARENA_MAX=2 keeps fragmentation bounded.
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"

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
elif [ "$METHOD" == "grpo_adaptivetemp" ]; then
    # AdamW baseline: bump temperature when entropy collapses.
    ADAPTIVE_TEMP=true
    TEMP_HIGH=${TEMP_HIGH:-1.2}
    LOW_ENT_RATIO=${LOW_ENT_RATIO:-0.5}
fi

if [ "$ADAPTIVE_TEMP" == "true" ]; then
    EXPNAME="${EXPNAME}-ADAPTTEMP_H${TEMP_HIGH}_R${LOW_ENT_RATIO}"
fi

if [ -n "$ESS_SCHEDULE" ] && [ "$ESS_SCHEDULE" != "constant" ]; then
    EXPNAME="${EXPNAME}-SCHED_${ESS_SCHEDULE}"
fi
if [ "$MIN_ESS" != "$ESS" ] && [ "$OPTIMIZER" == "ivon" ]; then
    EXPNAME="${EXPNAME}-MINESS_${MIN_ESS}"
fi
if [ "$M3PO_M" != 1 ]; then
    EXPNAME="${EXPNAME}-M3PO_M${M3PO_M}"
fi
if [ "$DECOUPLED_MC_SAMPLES" != 1 ]; then
    EXPNAME="${EXPNAME}-DECOUPLED${DECOUPLED_MC_SAMPLES}"
fi
if (( "$C3PO_N" > 1 )); then
    EXPNAME="${EXPNAME}-C3PO_N${C3PO_N}-seqmiss"
fi

# --- 3. DYNAMIC ARGUMENT CONSTRUCTION ---
# Handle KL_Cov/Clip_Cov logic
KL_COV_LINE=""
if [ "$KL_COV_RATIO" != "-1" ] && [ "$PPO_KL_COEF" != "-1" ]; then
    KL_COV_LINE="actor_rollout_ref.actor.policy_loss.kl_cov_ratio=$KL_COV_RATIO actor_rollout_ref.actor.policy_loss.ppo_kl_coef=$PPO_KL_COEF"
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
        actor_rollout_ref.actor.optim.ivon_config.m3po_m=$M3PO_M \
        actor_rollout_ref.actor.optim.ivon_config.decoupled_mc_samples=$DECOUPLED_MC_SAMPLES
    "
    if [ "$IVON_INIT_METHOD" == "trained" ]; then
        OPT_ARGS="${OPT_ARGS} \
            +actor_rollout_ref.actor.optim.optimizer_load_path=$MODEL_PATH
        "
    fi
else
    OPT_ARGS="actor_rollout_ref.actor.optim.optimizer=AdamW"
fi

LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-10}
EVAL_FREQ=${EVAL_FREQ:-0.1}
SAVE_FREQ=${SAVE_FREQ:-0.25}

# --- 4. PATHS & ENV ---
nnodes=1
nproc_per_node=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
project_name=VeRL-RL-Code

SAVE_ROOT=${SAVE_ROOT:-"${WORK}/bayesrl"}
SAVE_PATH=$SAVE_ROOT/$EXPNAME

# --- 5. EXECUTION ---
PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=$KL_COEF \
    algorithm.rollout_correction.rollout_is=sequence \
    algorithm.rollout_correction.rollout_rs=seq_sum_k1 \
    algorithm.rollout_correction.rollout_is_threshold=2 \
    algorithm.rollout_correction.rollout_is_batch_normalize=false \
    algorithm.rollout_correction.bypass_mode=true \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_prompt_length=$MAX_PROMPT_LEN \
    data.max_response_length=$MAX_RESPONSE_LEN \
    data.train_batch_size=32 \
    data.filter_overlong_prompts=true \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.adaptive_temperature=$ADAPTIVE_TEMP \
    actor_rollout_ref.rollout.temp_high=$TEMP_HIGH \
    actor_rollout_ref.rollout.low_ent_ratio=$LOW_ENT_RATIO \
    actor_rollout_ref.rollout.n=$GROUP_SIZE \
    actor_rollout_ref.rollout.c3po_n=$C3PO_N \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=3072 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.top_k=50 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    reward_model.sandbox_fusion.url=$SANDBOX_URL \
    reward_model.sandbox_fusion.max_concurrent=$SANDBOX_MAX_CONCURRENT \
    reward_model.sandbox_fusion.memory_limit_mb=$SANDBOX_MEMORY_LIMIT_MB \
    reward_model.sandbox_fusion.continuous=$SANDBOX_CONTINUOUS \
    reward_model.sandbox_fusion.timeout=$SANDBOX_TIMEOUT \
    reward_model.sandbox_fusion.retry_on_timeout=$SANDBOX_RETRY_ON_TIMEOUT \
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
