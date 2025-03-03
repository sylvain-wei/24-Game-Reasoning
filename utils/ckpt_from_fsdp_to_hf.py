import os
import argparse
import logging
from glob import glob
from collections import defaultdict
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# 参考指令
"""python ckpt_from_fsdp_to_hf.py --step 50 --fsdp_checkpoint_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math --huggingface_model_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math --output_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math_hf --world_size 2 --max_shard_size 10GB && python ckpt_from_fsdp_to_hf.py --step 100 --fsdp_checkpoint_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math --huggingface_model_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math --output_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math_hf --world_size 2 --max_shard_size 10GB && python ckpt_from_fsdp_to_hf.py --step 150 --fsdp_checkpoint_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math --huggingface_model_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math --output_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math_hf --world_size 2 --max_shard_size 10GB && python ckpt_from_fsdp_to_hf.py --step 200 --fsdp_checkpoint_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math --huggingface_model_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math --output_path /home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_sft_rl_v2/24game_qwen25_3b_math_hf --world_size 2 --max_shard_size 10GB"""


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="将FSDP checkpoint转换为Hugging Face格式")
    parser.add_argument("--step", type=str, required=True, help="模型的训练步数")
    parser.add_argument("--fsdp_checkpoint_path", type=str, required=True, help="FSDP checkpoint的基础路径")
    parser.add_argument("--huggingface_model_path", type=str, required=True, help="HuggingFace模型的路径")
    parser.add_argument("--output_path", type=str, required=True, help="输出模型的保存路径")
    parser.add_argument("--world_size", type=int, default=2, help="训练时的GPU数量")
    parser.add_argument("--max_shard_size", type=str, default="10GB", help="保存模型时每个分片的最大大小")
    return parser.parse_args()


def load_fsdp_checkpoints(fsdp_checkpoint_path: str, world_size: int) -> Dict[str, torch.Tensor]:
    """
    加载所有FSDP分片并将它们合并为完整的状态字典

    参数:
        fsdp_checkpoint_path: FSDP checkpoint文件所在路径
        world_size: 训练时使用的GPU数量

    返回:
        合并后的模型状态字典
    """
    state_dict = defaultdict(list)
    
    # 检查是否所有分片都存在
    missing_files = []
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        if not os.path.exists(filepath):
            missing_files.append(filepath)
    
    if missing_files:
        raise FileNotFoundError(f"无法找到以下FSDP checkpoint文件: {', '.join(missing_files)}")
    
    # 加载并合并所有分片
    for rank in range(world_size):
        filepath = f"{fsdp_checkpoint_path}/model_world_size_{world_size}_rank_{rank}.pt"
        logger.info(f"加载checkpoint: {filepath}")
        try:
            this_state_dict = torch.load(filepath, map_location='cpu')
            for key, value in this_state_dict.items():
                # 确保值可以转换为本地张量
                if hasattr(value, 'to_local'):
                    state_dict[key].append(value.to_local())
                else:
                    state_dict[key].append(value)
        except Exception as e:
            logger.error(f"加载checkpoint {filepath} 时出错: {str(e)}")
            raise
    
    # 沿着第0维连接所有分片
    for key in state_dict:
        try:
            state_dict[key] = torch.cat(state_dict[key], dim=0)
        except Exception as e:
            logger.error(f"合并key '{key}'的张量时出错: {str(e)}")
            raise
    
    return state_dict


def load_and_save_model(state_dict: Dict[str, torch.Tensor], 
                       huggingface_model_path: str, 
                       output_path: str,
                       max_shard_size: str = "10GB") -> None:
    """
    从状态字典加载模型并以Hugging Face格式保存

    参数:
        state_dict: 模型状态字典
        huggingface_model_path: HuggingFace模型路径(包含配置文件)
        output_path: 输出模型的保存路径
        max_shard_size: 保存模型时每个分片的最大大小
    """
    # 创建输出目录
    os.makedirs(output_path, exist_ok=True)
    
    try:
        # 加载模型配置
        logger.info(f"从 {huggingface_model_path} 加载模型配置")
        config = AutoConfig.from_pretrained(huggingface_model_path)
        
        # 从配置创建模型
        logger.info("根据配置初始化模型")
        model = AutoModelForCausalLM.from_config(config)
        
        # 将状态字典加载到模型中
        logger.info("将FSDP状态字典加载到模型中")
        model.load_state_dict(state_dict)
        
        # 保存模型
        logger.info(f"将模型保存到 {output_path}, 最大分片大小: {max_shard_size}")
        model.save_pretrained(output_path, max_shard_size=max_shard_size)
        
        # 加载并保存分词器
        logger.info(f"加载并保存分词器")
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model_path)
        tokenizer.save_pretrained(output_path)
        
        logger.info(f"模型和分词器已成功保存到 {output_path}")
    except Exception as e:
        logger.error(f"加载或保存模型时出错: {str(e)}")
        raise


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    try:
        # 构建完整路径
        step = args.step
        fsdp_checkpoint_path = f"{args.fsdp_checkpoint_path}/global_step_{step}/actor"
        huggingface_model_path = f"{args.huggingface_model_path}/global_step_{step}/actor/huggingface"
        output_path = f"{args.output_path}/checkpoint_global_step_{step}"
        
        logger.info(f"开始转换checkpoint: step={step}, world_size={args.world_size}")
        logger.info(f"FSDP路径: {fsdp_checkpoint_path}")
        logger.info(f"HuggingFace模型路径: {huggingface_model_path}")
        logger.info(f"输出路径: {output_path}")
        
        # 加载FSDP checkpoint
        state_dict = load_fsdp_checkpoints(fsdp_checkpoint_path, args.world_size)
        
        # 加载模型并保存为HuggingFace格式
        load_and_save_model(state_dict, huggingface_model_path, output_path, args.max_shard_size)
        
        logger.info("转换完成!")
    except Exception as e:
        logger.error(f"转换过程中出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()