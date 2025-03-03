import os
import json
import torch
import pandas as pd
import argparse
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 方法1：使用系统中已安装的中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'AR PL UMing CN']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 加载Qwen模型，对测试集进行评估
val_df_path = "data/24game_sft/val.parquet"


# qwen2.5-3B-Instruct的原始模型路径和四个epoch的checkpoints路径
model_original_path = "/home/weishaohang/workspace/models/models/Qwen/Qwen2.5-3B-Instruct"
# model_original_path = "/tmp/sft_model/global_step_364"
# model_epoch1_path = "/tmp/sft_model/global_step_91"
# model_epoch2_path = "/tmp/sft_model/global_step_182"
# model_epoch3_path = "/tmp/sft_model/global_step_273"
# model_epoch4_path = "/tmp/sft_model/global_step_364"

model_epoch1_path = "/home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_rl_zero/24game_qwen25_3b_math_hf/checkpoint_global_step_50"
model_epoch2_path = "/home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_rl_zero/24game_qwen25_3b_math_hf/checkpoint_global_step_100"
model_epoch3_path = "/home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_rl_zero/24game_qwen25_3b_math_hf/checkpoint_global_step_150"
model_epoch4_path = "/home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_rl_zero/24game_qwen25_3b_math_hf/checkpoint_global_step_200"
model_epoch5_path = "/home/weishaohang/workspace/24-Game-Reasoning/verl/checkpoints/verl_grpo_24game_rl_zero/24game_qwen25_3b_math_hf/checkpoint_global_step_250"


# 设置命令行参数
parser = argparse.ArgumentParser(description='评估24点游戏解题模型')
parser.add_argument('--base_model_path', type=str, default=model_original_path, help='基础模型路径')
parser.add_argument('--val_data_path', type=str, default=val_df_path, help='验证集路径')
parser.add_argument('--reference_json', type=str, default="./data/all_24_game_results_shuffled.json", help='参考结果JSON文件')
parser.add_argument('--max_length', type=int, default=8000, help='最大序列长度')
parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
parser.add_argument('--use_vllm', type=bool, default=True, help='是否使用VLLM进行推理')
parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU内存使用率')
parser.add_argument('--tensor_parallel_size', type=int, default=1, help='Tensor并行大小')
args = parser.parse_args()

def load_model_vllm(model_path):
    """使用VLLM加载模型"""
    print(f"使用VLLM加载模型: {model_path}")
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载VLLM模型
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        device="cuda", # 不设置也行
        dtype="bfloat16",
        enable_chunked_prefill=True,
        trust_remote_code=True
    )
    
    return llm, tokenizer

def load_model_transformers(model_path):
    """使用Transformers加载模型"""
    print(f"使用Transformers加载模型: {model_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(
        model_path, 
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer

def prepare_data(val_data_path, reference_json):
    """准备验证数据和参考结果"""
    # 加载验证集
    val_data = pd.read_parquet(val_data_path)
    
    # 加载参考结果
    with open(reference_json, 'r', encoding='utf-8') as f:
        reference_data = json.load(f)
    
    # 创建卡牌到可能性的映射
    card_to_possibility = {}
    for item in reference_data:
        if 'question' in item and 'is_possible' in item:
            # 从问题中提取卡片
            cards_match = re.search(r'Cards:([^.]+)', item['question'])
            if cards_match:
                cards = cards_match.group(1).strip()
                card_to_possibility[cards] = item['is_possible']
    
    # 为验证集数据添加参考答案
    val_data_with_reference = []
    for _, row in val_data.iterrows():
        prompt = row['prompt']
        # 从提示中提取卡片
        cards_match = re.search(r'Cards:([^.]+)', prompt)
        if cards_match:
            cards = cards_match.group(1).strip()
            if cards in card_to_possibility:
                val_data_with_reference.append({
                    'prompt': prompt,
                    'cards': cards,
                    'is_possible': card_to_possibility[cards]
                })
    
    print(f"已加载 {len(val_data_with_reference)} 条验证数据")
    return val_data_with_reference

def generate_response_vllm(llm, tokenizer, prompts, max_length=8000):
    """使用VLLM批量生成模型响应"""
    # 设置采样参数
    sampling_params = SamplingParams(
        n=1,  # 每个prompt生成1个输出
        temperature=0.7,
        max_tokens=max_length,
        stop=[tokenizer.eos_token],  # 生成到eos_token就停止
    )
    
    # 批量生成响应
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    
    # 提取生成的文本
    responses = []
    for output in outputs:
        # 获取生成的文本并去除提示部分
        response_text = output.outputs[0].text
        prompt_text = output.prompt
        # 去除提示部分，只保留回答
        response = response_text.strip()
        responses.append(response)
    
    return responses

def generate_response_transformers(model, tokenizer, prompt, max_length=8000):
    """使用Transformers生成单个模型响应"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,
            temperature=0.7,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 去除提示部分，只保留回答
    response = response.replace(prompt, "").strip()
    return response

def extract_answer(response):
    """从模型响应中提取答案"""
    # 尝试从回答中提取 <answer>...</answer> 内容
    answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        return answer
    
    # 如果没有标签，查找是否直接以公式或 NO 开始
    equation_match = re.search(r'^([^<>(]*?)\s*=\s*24', response)
    if equation_match:
        return equation_match.group(0).strip()
    
    if "NO" in response[:50]:  # 检查开头是否包含 NO
        return "NO"
    
    # 如果无法匹配，返回原始响应的前100个字符
    return response[:100]

def extract_numbers_from_expression(expression):
    """从表达式中提取所有数字"""
    # 使用正则表达式匹配所有数字
    numbers = re.findall(r'\d+', expression)
    # 将字符串转换为整数
    numbers = [int(num) for num in numbers if num != "24"]
    return numbers

def compute_equation(equation_str, cards_str):
    """计算等式是否成立，验证结果是否为24，并检查使用的数字是否与卡片匹配"""
    try:
        # 去掉等号后面的部分，仅计算等号前面的表达式
        if "=" in equation_str:
            expression = equation_str.split("=")[0].strip()
        else:
            expression = equation_str.strip()
        
        # 替换中文符号为英文符号
        expression = expression.replace('×', '*').replace('÷', '/')
        
        # 处理括号不匹配的情况
        open_brackets = expression.count('(')
        close_brackets = expression.count(')')
        if open_brackets > close_brackets:
            expression += ')' * (open_brackets - close_brackets)
        
        # 提取表达式中的所有数字
        equation_numbers = extract_numbers_from_expression(expression)
        
        # 从卡片字符串中提取卡片数字
        cards = [int(num.strip()) for num in cards_str.split(',')]
        
        # 检查两组数字是否使用了相同的数字
        if sorted(equation_numbers) != sorted(cards):
            return False
        
        # 计算表达式的值
        result = eval(expression)
        
        # 考虑浮点数精度问题，允许小范围误差
        return abs(result - 24) < 1e-6
    except Exception as e:
        # 如果计算出错，返回False
        return False

def evaluate_answers(responses, val_data):
    """评估模型回答的准确率"""
    correct = 0
    incorrect = 0
    
    # 存储详细结果
    details = []
    
    for i, (response, data) in enumerate(zip(responses, val_data)):
        answer = extract_answer(response)
        
        # 判断答案是否正确
        if data['is_possible']:
            # 如果有解，应该返回方程式，而不是 NO
            has_equation = "=" in answer
            if has_equation:
                # 验证算式计算结果是否为24，并且使用了正确的卡片数字
                is_correct = compute_equation(answer, data["cards"])
            else:
                is_correct = False
        else:
            # 如果无解，应该返回 NO
            is_correct = answer.strip() == "NO"
        
        if is_correct:
            correct += 1
        else:
            incorrect += 1
        
        details.append({
            "cards": data["cards"],
            "is_possible": data["is_possible"],
            "is_correct": is_correct,
            "answer": answer,
            "response": response,
        })
    
    accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "incorrect": incorrect,
        "total": correct + incorrect,
        "details": details
    }

def evaluate_model_vllm(model_path, val_data):
    """使用VLLM评估模型"""
    # 加载模型和分词器
    llm, tokenizer = load_model_vllm(model_path)
    
    # 提取所有prompt
    prompts = [data["prompt"] for data in val_data]
    
    # 批量处理，分批生成响应
    batch_size = args.batch_size
    all_responses = []
    
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = generate_response_vllm(llm, tokenizer, batch_prompts, args.max_length)
        all_responses.extend(batch_responses)
    
    # 评估结果
    results = evaluate_answers(all_responses, val_data)
    
    print(f"模型准确率: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    
    return results

def evaluate_model_transformers(model_path, val_data):
    """使用Transformers评估模型"""
    # 加载模型和分词器
    model, tokenizer = load_model_transformers(model_path)
    
    # 逐个生成响应
    responses = []
    for data in tqdm(val_data):
        response = generate_response_transformers(model, tokenizer, data["prompt"], args.max_length)
        responses.append(response)
    
    # 评估结果
    results = evaluate_answers(responses, val_data)
    
    print(f"模型准确率: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    
    return results

def main():
    print("开始评估模型...")
    
    # 准备验证数据
    val_data = prepare_data(args.val_data_path, args.reference_json)
    
    # 确定评估函数
    evaluation_func = evaluate_model_vllm if args.use_vllm else evaluate_model_transformers
    
    # 评估基础模型
    print(f"评估基础模型: {args.base_model_path}")
    base_results = evaluation_func(args.base_model_path, val_data)
    
    # 定义checkpoint路径列表
    checkpoint_paths = [    # TODO:修改训练轮数
        (50, model_epoch1_path),
        (100, model_epoch2_path),
        (150, model_epoch3_path),
        (200, model_epoch4_path),
        (250, model_epoch5_path)
    ]
    
    checkpoint_results = []
    
    # 评估每个checkpoint
    for epoch, checkpoint_path in checkpoint_paths:
        if not os.path.exists(checkpoint_path):
            print(f"跳过不存在的Checkpoint: {checkpoint_path}")
            continue
            
        print(f"评估Checkpoint Epoch {epoch}: {checkpoint_path}")
        
        # 评估checkpoint
        results = evaluation_func(checkpoint_path, val_data)
        
        # 存储结果
        checkpoint_results.append({
            "epoch": epoch,
            "accuracy": results['accuracy'],
            "correct": results['correct'],
            "total": results['total'],
            "details": results['details']
        })
    
    # 保存完整评估结果
    combined_results = {
        "base_model": {
            "accuracy": base_results['accuracy'],
            "correct": base_results['correct'],
            "total": base_results['total'],
            "details": base_results['details']
        },
        "checkpoints": checkpoint_results
    }
    
    # 保存结果到文件
    with open("evaluation_results_sft_qwen_2.5_3b_instruct.json", "w", encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2)
    
    # 绘制准确率变化图
    epochs = [0] + [r["epoch"] for r in checkpoint_results]
    accuracies = [base_results["accuracy"]] + [r["accuracy"] for r in checkpoint_results]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, accuracies, marker='o', linestyle='-')
    plt.title('24点游戏解题准确率随训练轮次的变化')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率')
    plt.xticks(epochs)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    for i, (x, y) in enumerate(zip(epochs, accuracies)):
        plt.annotate(f"{y:.4f}", (x, y), textcoords="offset points", 
                    xytext=(0, 10), ha='center')
    
    plt.tight_layout()
    plt.savefig("accuracy_trend_sft_qwen_2.5_3b_instruct.png")
    plt.show()
    
    print("评估完成，结果已保存到 evaluation_results.json 和 accuracy_trend.png")

if __name__ == "__main__":
    main()

