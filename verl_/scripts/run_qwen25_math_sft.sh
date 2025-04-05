#!/bin/bash
set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen25_math_sft.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2
# cd verl
# bash scripts/run_qwen25_math_sft.sh 4 None
# CUDA_VISIBLE_DEVICES=0,1,3 bash scripts/run_qwen25_math_sft.sh 3 None

shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_batch_size=18 \
    data.micro_batch_size_per_gpu=3 \
    "+data.prompt_dict_keys=['question']" \
    "+data.response_dict_keys=['completion']" \
    data.max_length=4096 \
    data.truncation="right" \
    model.enable_gradient_checkpointing=True \
    model.use_liger=False \
    optim.lr=5e-6 \
    optim.warmup_steps_ratio=0.1 \
    trainer.project_name=qwen25-24game-sft \
    trainer.experiment_name=qwen25-3b-24game-sft \
    "trainer.logger=['console','wandb']" \
    trainer.default_hdfs_dir=null $@