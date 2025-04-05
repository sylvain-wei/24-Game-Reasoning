#!/bin/bash
set -x  # 启用调试模式，将执行的每个命令打印到终端

if [ "$#" -lt 2 ]; then  # 检查参数数量是否少于2个
    echo "Usage: run_qwen25_math_sft.sh <nproc_per_node> <save_path> [other_configs...]"  # 打印使用说明
    exit 1  # 如果参数不足则退出脚本
fi

nproc_per_node=$1  # 将第一个参数赋值给nproc_per_node（每个节点的进程数）
save_path=$2  # 将第二个参数赋值给save_path（保存路径）

# Shift the arguments so $@ refers to the rest  # 移动参数，使$@引用剩余的参数
shift 2  # 移除前两个参数

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \  # 使用torchrun启动分布式训练，单节点模式，指定每节点进程数
     -m verl.trainer.fsdp_sft_trainer \
     # 使用FSDP（全分片数据并行）监督微调训练器模块
    # data.train_files=$HOME/data/math/train.parquet \  # 设置训练数据文件路径
    # data.val_files=$HOME/data/math/test.parquet \  # 设置验证数据文件路径
    # data.prompt_key=extra_info \  # 设置提示信息的键名为extra_info
    # data.response_key=extra_info \  # 设置响应信息的键名为extra_info
    "+data.prompt_dict_keys=['question']" \
    # 添加提示词典键为question
    "+data.response_dict_keys=['answer']" \
    # 添加响应词典键为answer
    data.train_batch_size=16 \
    # 设置训练批次大小为16
    data.micro_batch_size_per_gpu=4 \
    # 设置每个GPU的微批次大小为4
    data.max_length=2048 \
    # 设置最大长度为2048
    # model.partial_pretrain=Qwen/Qwen2.5-7B-Instruct \  # 使用Qwen2.5-7B-Instruct预训练模型
    # model.trust_remote_code=True \  # 信任远程代码
    model.enable_gradient_checkpointing=True \
    # 启用梯度检查点
    model.fsdp_config.wrap_policy.block_layer_name=QWenBlock \
    # 设置FSDP配置的块层名称为QWenBlock
    model.use_liger=False \
    # 不使用Liger
    optim.lr=2e-5 \
    # 设置学习率为2e-5
    # optim.weight_decay=0.0 \  # 设置权重衰减为0.0
    optim.warmup_steps_ratio=0.03 \
    # 设置预热步数比例为0.03
    # trainer.default_local_dir=$save_path \  # 设置训练器的默认本地目录为保存路径
    trainer.project_name=qwen25-24game-sft \
    # 设置项目名称为qwen25-math-sft
    trainer.experiment_name=qwen25-7b-24game-sft \
    # 设置实验名称为qwen25-7b-math-sft
    # trainer.total_epochs=3 \  # 设置总训练轮数为3
    # ulysses_sequence_parallel_size=1 \  # 设置Ulysses序列并行大小为1
    # use_remove_padding=False \  # 不使用移除填充
    "trainer.logger=['console','wandb']" \
    # 设置日志输出到控制台和wandb
    trainer.default_hdfs_dir=null $@
    # 设置默认HDFS目录为null，并传递剩余的命令行参数