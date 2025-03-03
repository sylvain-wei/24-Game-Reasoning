import os
import json
import pandas as pd
import random
from pathlib import Path
import argparse
import numpy as np

def load_json_data(json_path):
    """加载JSON格式的24点游戏结果数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_game_data(data_item):
    """处理单个游戏数据条目为SFT训练格式"""
    # 构造输入提示
    prompt = data_item["prompt"].split("Assistant:")[0].strip()
    
    # 构造目标输出(completion)
    # 使用<think>和<answer>标签包装推理和答案
    completion = f"<think> {data_item['reasoning_content']} </think><answer> {data_item['answer']} </answer>"
    
    return {
        "prompt": prompt,
        "completion": completion,
        "is_possible": data_item.get("is_possible", None),
        "question": data_item.get("question", ""),
        "answer": data_item.get("answer", "")
    }

def prepare_dataset(json_path, output_dir, train_ratio=0.9, seed=42):
    """准备训练和验证数据集"""
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    data = load_json_data(json_path)
    print(f"Loaded {len(data)} items from {json_path}")
    
    # 随机打乱数据
    random.shuffle(data)
    
    # 划分训练集和验证集
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # 处理数据为SFT格式
    train_processed = [process_game_data(item) for item in train_data]
    val_processed = [process_game_data(item) for item in val_data]
    
    # 转换为DataFrame并保存为Parquet
    train_df = pd.DataFrame(train_processed)
    val_df = pd.DataFrame(val_processed)
    
    train_output = os.path.join(output_dir, "train.parquet")
    val_output = os.path.join(output_dir, "val.parquet")
    
    train_df.to_parquet(train_output, index=False)
    val_df.to_parquet(val_output, index=False)
    
    print(f"Saved {len(train_processed)} train samples to {train_output}")
    print(f"Saved {len(val_processed)} validation samples to {val_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare 24 Game data for SFT")
    parser.add_argument("--input", type=str,
                        default="data/all_24_game_results_shuffled.json",
                        help="Path to the shuffled 24 game JSON file")
    parser.add_argument("--output_dir", type=str, 
                        default="data/24game_sft",
                        help="Directory to save the processed datasets")
    parser.add_argument("--train_ratio", type=float, default=0.9, help="Ratio of data for training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    prepare_dataset(
        json_path=args.input,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
