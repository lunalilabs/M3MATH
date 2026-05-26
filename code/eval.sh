#!/bin/bash

set -euo pipefail
set -x

DATA_DIR="data/m3math"  # data directory
MODEL_PATH="model/Qwen/Qwen3-4B"  # model directory
OUTPUT_ROOT="eval_outputs/qwen3_4b"  # output root directory
RUN_NAME="qwen3_4b_eval"  # run name for logging
REWARD_PATH="reward.py"  # reward function file

# data paths
train_path="$DATA_DIR/train.parquet"
test_path="$DATA_DIR/test.parquet"

train_files="['$train_path']"
test_files="['$test_path']"

mkdir -p "$OUTPUT_ROOT"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=8 \
    data.val_batch_size=4 \
    data.val_max_samples=500 \
    data.shuffle=False \
    data.seed=42 \
    data.validation_shuffle=False \
    data.max_prompt_length=204800 \
    data.max_response_length=204800 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    critic.enable=False \
    reward.num_workers=4 \
    reward.reward_manager.name=naive \
    reward.reward_model.enable=False \
    reward.custom_reward_function.path="$REWARD_PATH" \
    reward.custom_reward_function.name=compute_score \
    algorithm.use_kl_in_reward=False \
    trainer.project_name=verl_eval \
    trainer.experiment_name="$RUN_NAME" \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.total_epochs=1 \
    trainer.validation_data_dir="$OUTPUT_ROOT/$RUN_NAME" \
    "${extra_args[@]}" \
    "$@"
