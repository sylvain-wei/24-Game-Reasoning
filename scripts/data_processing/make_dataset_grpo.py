"""
处理24点游戏数据集，为GRPO训练准备数据
"""
import os
import json
import argparse
import random
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
import numpy as np

# 可选：如果需要使用HDFS
try:
    from verl.utils.hdfs_io import copy, makedirs
    HDFS_AVAILABLE = True
except ImportError:
    HDFS_AVAILABLE = False
    print("HDFS相关功能不可用，仅支持本地存储")

def parse_args():
    parser = argparse.ArgumentParser(description="处理24点游戏数据集，为GRPO训练准备数据")
    parser.add_argument("--data_path", type=str, 
                        default="/home/weishaohang/workspace/24-Game-Reasoning/data/all_24_game_results_shuffled.json", 
                        help="原始数据集路径")
    parser.add_argument("--output_dir", type=str, 
                        default="data/24game_grpo", 
                        help="输出数据目录")
    parser.add_argument("--hdfs_dir", type=str, 
                        default=None, 
                        help="HDFS目录，如果不需要则不设置")
    parser.add_argument("--train_ratio", type=float, 
                        default=0.9, 
                        help="训练集占比")
    parser.add_argument("--seed", type=int, 
                        default=42, 
                        help="随机种子")
    return parser.parse_args()

def format_prompt_content(example):
    """构造模型输入的prompt内容格式"""
    prompt_content = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. User: {example['question']} Assistant:"
    return prompt_content

def process_data(data_list):
    """处理数据，转换为适合GRPO训练的格式"""
    processed_data = []
    for idx, item in tqdm(enumerate(data_list), desc="处理数据", total=len(data_list)):
        # 构造数据项
        data_item = {
            "prompt_content": format_prompt_content(item),
            "reasoning_content": item.get("reasoning_content", ""),
            "answer": item.get("answer", ""),
            "result": item.get("result", ""),
            "question": item.get("question", ""),
            "is_possible": item.get("is_possible", False)
        }
        processed_data.append(data_item)
    return processed_data

def main():
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载原始数据
    print(f"加载数据: {args.data_path}")
    with open(args.data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 处理数据
    processed_data = process_data(data)
    
    # 划分训练集和测试集
    split_idx = int(len(processed_data) * args.train_ratio)
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]
    
    print(f"训练集大小: {len(train_data)}")
    print(f"测试集大小: {len(test_data)}")
    
    # 转换为Dataset格式
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    test_dataset = Dataset.from_pandas(pd.DataFrame(test_data))
    
    # 为GRPO训练格式化数据
    def format_for_grpo(example, idx, split):
        # 将prompt改为dict格式，与arth.py保持一致
        data = {
            "data_source": "24game",
            "prompt": [{
                "role": "user",
                "content": example["prompt_content"],
            }],
            "ability": "reasoning",
            "reward_model": {
                "style": "24game_rule", # 修改为自定义reward评估方式
                "ground_truth": example["answer"] if example["is_possible"] else "NO",
                "is_possible": example["is_possible"] # 添加可解性信息
            },
            "extra_info": {
                'split': split,
                'index': idx,
                'reasoning_content': example["reasoning_content"],
                'result': example["result"],
                'question': example["question"],
                'is_possible': example["is_possible"]
            }
        }
        return data
    
    # 应用格式化
    train_dataset = train_dataset.map(
        lambda example, idx: format_for_grpo(example, idx, "train"),
        with_indices=True
    )
    test_dataset = test_dataset.map(
        lambda example, idx: format_for_grpo(example, idx, "val"),
        with_indices=True
    )
    
    # 保存数据
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "val.parquet")
    
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)
    
    print(f"训练数据已保存到: {train_path}")
    print(f"测试数据已保存到: {test_path}")
    
    # 如果需要保存到HDFS
    if args.hdfs_dir and HDFS_AVAILABLE:
        print(f"复制数据到HDFS: {args.hdfs_dir}")
        makedirs(args.hdfs_dir)
        copy(src=output_dir, dst=args.hdfs_dir)
        print(f"数据已复制到HDFS: {args.hdfs_dir}")

if __name__ == "__main__":
    main()
