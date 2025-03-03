import json
import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 方法1：使用系统中已安装的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'AR PL UMing CN']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def load_evaluation_results(file_path):
    """从JSON文件加载评估结果数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_thinking_chain_length(response):
    """提取思维链长度"""
    # 查找<think>和</think>标记
    thinking_pattern = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    
    if thinking_pattern:
        # 如果找到了完整的思维链，返回其长度
        return len(thinking_pattern.group(1))
    else:
        # 如果找不到完整的思维链，返回整个响应的长度
        return len(response)

def has_complete_thinking_chain(response):
    """检查响应中是否包含完整的思维链"""
    # 查找<think>和</think>标记
    thinking_pattern = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
    return thinking_pattern is not None

def calculate_avg_thinking_length(details):
    """计算平均思维链长度"""
    total_length = 0
    count = 0
    
    for detail in details:
        response = detail.get('response', '')
        if response:
            length = extract_thinking_chain_length(response)
            total_length += length
            count += 1
    
    if count > 0:
        return total_length / count
    return 0

def calculate_complete_thinking_stats(details):
    """计算仅包含完整思维链的案例的统计数据"""
    total_length = 0
    complete_count = 0
    correct_count = 0
    
    # 过滤出有完整思维链的案例
    complete_cases = []
    for detail in details:
        response = detail.get('response', '')
        if response and has_complete_thinking_chain(response):
            length = extract_thinking_chain_length(response)
            total_length += length
            complete_count += 1
            
            if detail.get('is_correct', False):
                correct_count += 1
            
            complete_cases.append(detail)
    
    # 计算平均长度和准确率
    avg_length = total_length / complete_count if complete_count > 0 else 0
    accuracy = correct_count / complete_count if complete_count > 0 else 0
    
    return {
        'avg_length': avg_length,
        'accuracy': accuracy,
        'complete_count': complete_count,
        'correct_count': correct_count,
        'complete_cases': complete_cases
    }

def main():
    # 加载评估结果数据
    file_path = '/home/weishaohang/workspace/24-Game-Reasoning/evaluation_results_rl_zero_v1.json'    # TODO:
    results = load_evaluation_results(file_path)
    
    # 初始化数据存储 - 所有案例
    model_versions = ['Base Model']
    thinking_lengths = []
    accuracies = []
    
    # 初始化数据存储 - 仅完整思维链案例
    complete_model_versions = ['Base Model']
    complete_thinking_lengths = []
    complete_accuracies = []
    complete_counts = []
    
    # 处理基础模型
    base_details = results['base_model'].get('details', [])
    base_accuracy = results['base_model'].get('accuracy', 0)
    base_avg_thinking_length = calculate_avg_thinking_length(base_details)
    
    # 计算完整思维链的统计数据
    base_complete_stats = calculate_complete_thinking_stats(base_details)
    
    thinking_lengths.append(base_avg_thinking_length)
    accuracies.append(base_accuracy)
    
    complete_thinking_lengths.append(base_complete_stats['avg_length'])
    complete_accuracies.append(base_complete_stats['accuracy'])
    complete_counts.append(base_complete_stats['complete_count'])
    
    # 处理每个checkpoint
    for i, checkpoint in enumerate(results.get('checkpoints', [])):
        model_version = f'Checkpoint {i+1}'
        model_versions.append(model_version)
        complete_model_versions.append(model_version)
        
        checkpoint_details = checkpoint.get('details', [])
        checkpoint_accuracy = checkpoint.get('accuracy', 0)
        checkpoint_avg_thinking_length = calculate_avg_thinking_length(checkpoint_details)
        
        # 计算完整思维链的统计数据
        checkpoint_complete_stats = calculate_complete_thinking_stats(checkpoint_details)
        
        thinking_lengths.append(checkpoint_avg_thinking_length)
        accuracies.append(checkpoint_accuracy)
        
        complete_thinking_lengths.append(checkpoint_complete_stats['avg_length'])
        complete_accuracies.append(checkpoint_complete_stats['accuracy'])
        complete_counts.append(checkpoint_complete_stats['complete_count'])
    
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # 绘制所有案例的思维链长度与准确率的关系图
    ax1.plot(thinking_lengths, accuracies, marker='o', linestyle='-', color='blue')
    
    # 添加数据标签
    for i, (length, acc, version) in enumerate(zip(thinking_lengths, accuracies, model_versions)):
        ax1.annotate(f'{version}\n({length:.1f}, {acc:.4f})', 
                    (length, acc), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=9)
    
    # 设置第一个子图属性
    ax1.set_title('所有案例：平均思维链长度与准确率的关系')
    ax1.set_xlabel('平均思维链长度 (字符数)')
    ax1.set_ylabel('准确率')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # 绘制仅完整思维链案例的思维链长度与准确率的关系图
    ax2.plot(complete_thinking_lengths, complete_accuracies, marker='o', linestyle='-', color='green')
    
    # 添加数据标签
    for i, (length, acc, version, count) in enumerate(zip(complete_thinking_lengths, complete_accuracies, complete_model_versions, complete_counts)):
        ax2.annotate(f'{version}\n({length:.1f}, {acc:.4f})\n案例数: {count}', 
                    (length, acc), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=9)
    
    # 设置第二个子图属性
    ax2.set_title('仅完整思维链案例：平均思维链长度与准确率的关系')
    ax2.set_xlabel('平均思维链长度 (字符数)')
    ax2.set_ylabel('准确率')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig('thinking_length_vs_accuracy_comparison.png', dpi=300)
    plt.show()
    
    # 打印所有案例的统计数据
    print("\n所有案例的思维链长度与准确率统计结果:")
    print("-" * 60)
    print(f"{'模型版本':<15} {'平均思维链长度':>20} {'准确率':>10}")
    print("-" * 60)
    
    for version, length, acc in zip(model_versions, thinking_lengths, accuracies):
        print(f"{version:<15} {length:>20.2f} {acc:>10.4f}")
    
    # 打印仅完整思维链案例的统计数据
    print("\n仅完整思维链案例的统计结果:")
    print("-" * 75)
    print(f"{'模型版本':<15} {'平均思维链长度':>20} {'准确率':>10} {'完整思维链案例数':>15} {'占比(%)':<10}")
    print("-" * 75)
    
    for i, (version, length, acc, count) in enumerate(zip(complete_model_versions, complete_thinking_lengths, complete_accuracies, complete_counts)):
        # 计算完整案例占总案例的比例
        if i == 0:  # 基础模型
            total_count = len(results['base_model']['details'])
        else:  # checkpoint模型
            total_count = len(results['checkpoints'][i-1]['details'])
        
        percentage = (count / total_count * 100) if total_count > 0 else 0
        print(f"{version:<15} {length:>20.2f} {acc:>10.4f} {count:>15} {percentage:>10.2f}")

if __name__ == "__main__":
    main()
