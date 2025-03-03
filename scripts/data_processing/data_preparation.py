# 富集采样R1生成的long-cot数据

from utils import *
import random
import itertools
import math
import numpy as np
from tqdm import tqdm
import json
import os
import re

def is_possible_24(nums):
    """
    检查给定的4个数字是否可能通过加减乘除得到24
    使用暴力方法检查所有可能的运算组合
    """
    ops = ['+', '-', '*', '/']
    
    # 生成所有可能的数字排列
    for nums_perm in itertools.permutations(nums):
        # 生成所有可能的运算符组合
        for op1 in ops:
            for op2 in ops:
                for op3 in ops:  # 修复了这一行，添加了 in ops
                    # 尝试不同的括号组合
                    
                    # ((a op1 b) op2 c) op3 d
                    try:
                        if op1 == '+':
                            res1 = nums_perm[0] + nums_perm[1]
                        elif op1 == '-':
                            res1 = nums_perm[0] - nums_perm[1]
                        elif op1 == '*':
                            res1 = nums_perm[0] * nums_perm[1]
                        else:  # 除法
                            if nums_perm[1] == 0:  # 避免除以零
                                continue
                            res1 = nums_perm[0] / nums_perm[1]
                            
                        if op2 == '+':
                            res2 = res1 + nums_perm[2]
                        elif op2 == '-':
                            res2 = res1 - nums_perm[2]
                        elif op2 == '*':
                            res2 = res1 * nums_perm[2]
                        else:  # 除法
                            if nums_perm[2] == 0:  # 避免除以零
                                continue
                            res2 = res1 / nums_perm[2]
                            
                        if op3 == '+':
                            res3 = res2 + nums_perm[3]
                        elif op3 == '-':
                            res3 = res2 - nums_perm[3]
                        elif op3 == '*':
                            res3 = res2 * nums_perm[3]
                        else:  # 除法
                            if nums_perm[3] == 0:  # 避免除以零
                                continue
                            res3 = res2 / nums_perm[3]
                            
                        if abs(res3 - 24) < 1e-10:
                            return True
                    except Exception as e:
                        # 明确处理异常
                        continue
                    
                    # (a op1 (b op2 c)) op3 d
                    try:
                        if op2 == '+':
                            res1 = nums_perm[1] + nums_perm[2]
                        elif op2 == '-':
                            res1 = nums_perm[1] - nums_perm[2]
                        elif op2 == '*':
                            res1 = nums_perm[1] * nums_perm[2]
                        else:  # 除法
                            if nums_perm[2] == 0:  # 避免除以零
                                continue
                            res1 = nums_perm[1] / nums_perm[2]
                            
                        if op1 == '+':
                            res2 = nums_perm[0] + res1
                        elif op1 == '-':
                            res2 = nums_perm[0] - res1
                        elif op1 == '*':
                            res2 = nums_perm[0] * res1
                        else:  # 除法
                            if res1 == 0:  # 避免除以零
                                continue
                            res2 = nums_perm[0] / res1
                            
                        if op3 == '+':
                            res3 = res2 + nums_perm[3]
                        elif op3 == '-':
                            res3 = res2 - nums_perm[3]
                        elif op3 == '*':
                            res3 = res2 * nums_perm[3]
                        else:  # 除法
                            if nums_perm[3] == 0:  # 避免除以零
                                continue
                            res3 = res2 / nums_perm[3]
                            
                        if abs(res3 - 24) < 1e-10:
                            return True
                    except:
                        continue
                    
                    # a op1 ((b op2 c) op3 d)
                    try:
                        if op2 == '+':
                            res1 = nums_perm[1] + nums_perm[2]
                        elif op2 == '-':
                            res1 = nums_perm[1] - nums_perm[2]
                        elif op2 == '*':
                            res1 = nums_perm[1] * nums_perm[2]
                        else:  # 除法
                            if nums_perm[2] == 0:  # 避免除以零
                                continue
                            res1 = nums_perm[1] / nums_perm[2]
                            
                        if op3 == '+':
                            res2 = res1 + nums_perm[3]
                        elif op3 == '-':
                            res2 = res1 - nums_perm[3]
                        elif op3 == '*':
                            res2 = res1 * nums_perm[3]
                        else:  # 除法
                            if nums_perm[3] == 0:  # 避免除以零
                                continue
                            res2 = res1 / nums_perm[3]
                            
                        if op1 == '+':
                            res3 = nums_perm[0] + res2
                        elif op1 == '-':
                            res3 = nums_perm[0] - res2
                        elif op1 == '*':
                            res3 = nums_perm[0] * res2
                        else:  # 除法
                            if res2 == 0:  # 避免除以零
                                continue
                            res3 = nums_perm[0] / res2
                            
                        if abs(res3 - 24) < 1e-10:
                            return True
                    except:
                        continue
                    
                    # a op1 (b op2 (c op3 d))
                    try:
                        if op3 == '+':
                            res1 = nums_perm[2] + nums_perm[3]
                        elif op3 == '-':
                            res1 = nums_perm[2] - nums_perm[3]
                        elif op3 == '*':
                            res1 = nums_perm[2] * nums_perm[3]
                        else:  # 除法
                            if nums_perm[3] == 0:  # 避免除以零
                                continue
                            res1 = nums_perm[2] / nums_perm[3]
                            
                        if op2 == '+':
                            res2 = nums_perm[1] + res1
                        elif op2 == '-':
                            res2 = nums_perm[1] - res1
                        elif op2 == '*':
                            res2 = nums_perm[1] * res1
                        else:  # 除法
                            if res1 == 0:  # 避免除以零
                                continue
                            res2 = nums_perm[1] / res1
                            
                        if op1 == '+':
                            res3 = nums_perm[0] + res2
                        elif op1 == '-':
                            res3 = nums_perm[0] - res2
                        elif op1 == '*':
                            res3 = nums_perm[0] * res2
                        else:  # 除法
                            if res2 == 0:  # 避免除以零
                                continue
                            res3 = nums_perm[0] / res2
                            
                        if abs(res3 - 24) < 1e-10:
                            return True
                    except:
                        continue
                    
                    # (a op1 b) op2 (c op3 d)
                    try:
                        if op1 == '+':
                            res1 = nums_perm[0] + nums_perm[1]
                        elif op1 == '-':
                            res1 = nums_perm[0] - nums_perm[1]
                        elif op1 == '*':
                            res1 = nums_perm[0] * nums_perm[1]
                        else:  # 除法
                            if nums_perm[1] == 0:  # 避免除以零
                                continue
                            res1 = nums_perm[0] / nums_perm[1]
                        
                        if op3 == '+':
                            res2 = nums_perm[2] + nums_perm[3]
                        elif op3 == '-':
                            res2 = nums_perm[2] - nums_perm[3]
                        elif op3 == '*':
                            res2 = nums_perm[2] * nums_perm[3]
                        else:  # 除法
                            if nums_perm[3] == 0:  # 避免除以零
                                continue
                            res2 = nums_perm[2] / nums_perm[3]
                        
                        if op2 == '+':
                            res3 = res1 + res2
                        elif op2 == '-':
                            res3 = res1 - res2
                        elif op2 == '*':
                            res3 = res1 * res2
                        else:  # 除法
                            if res2 == 0:  # 避免除以零
                                continue
                            res3 = res1 / res2
                        
                        if abs(res3 - 24) < 1e-10:
                            return True
                    except:
                        continue
    
    # 如果所有组合都尝试过了，仍然无法得到24，则返回False
    return False

def gen_not_24_inputs(num_samples=300000, max_val=13, output_file="not_24_inputs.txt"):
    """
    暴力遍历生成不可能构造出24的数字组合
    每组包含4个数字，范围从1到max_val，允许重复数字
    
    :param num_samples: 要生成的样本数量上限
    :param max_val: 数字的最大值
    :param output_file: 输出文件名
    :return: 生成的数据列表
    """
    print(f"开始生成不可能构造出24的数字组合...")
    
    results = []
    not_24_count = 0
    unique_combs = set()
    
    # 创建文件，清空之前的内容
    with open(output_file, "w") as f:
        pass
    
    # 生成所有可能的4个数字组合（包括重复数字）
    all_combinations = list(itertools.combinations_with_replacement(range(1, max_val+1), 4))
    random.shuffle(all_combinations)  # 随机打乱以避免偏向性
    
    print(f"共生成了{len(all_combinations)}种可能的数字组合")
    
    with tqdm(total=min(num_samples, len(all_combinations))) as pbar:
        for combination in all_combinations:
            if not_24_count >= num_samples:
                break
                
            nums = list(combination)
            
            # 确保组合是唯一的
            nums_tuple = tuple(sorted(nums))
            if nums_tuple in unique_combs:
                continue
            
            # 验证是否确实不能构造出24
            if not is_possible_24(nums):
                results.append(nums)
                unique_combs.add(nums_tuple)
                not_24_count += 1
                pbar.update(1)
                
                # 直接写入文件
                with open(output_file, "a") as f:
                    f.write(" ".join(map(str, nums)) + "\n")
                
                # 每生成10000个打印进度信息
                if not_24_count % 10000 == 0:
                    print(f"已生成 {not_24_count} 组数据")
    
    print(f"成功生成并保存了{not_24_count}组不可能构造出24的数字组合到{output_file}")
    return results

def gen_all_24_inputs(max_val=13, output_file="24_inputs.txt"):
    """
    暴力遍历生成所有可以构造出24的数字组合
    每组包含4个数字，范围从1到max_val，允许重复数字
    
    :param max_val: 数字的最大值
    :param output_file: 输出文件名
    :return: 生成的数据列表
    """
    print(f"开始生成所有可以构造出24的数字组合（数值范围1-{max_val}）...")
    
    results = []
    valid_24_count = 0
    unique_combs = set()
    
    # 创建文件，清空之前的内容
    with open(output_file, "w") as f:
        pass
    
    # 生成所有可能的4个数字组合（包括重复数字）
    all_combinations = list(itertools.combinations_with_replacement(range(1, max_val+1), 4))
    print(f"共生成了{len(all_combinations)}种可能的数字组合")
    
    with tqdm(total=len(all_combinations)) as pbar:
        for combination in all_combinations:
            nums = list(combination)
            
            # 确保组合是唯一的
            nums_tuple = tuple(sorted(nums))
            if nums_tuple in unique_combs:
                continue
                
            unique_combs.add(nums_tuple)
            
            # 验证是否可以构造出24
            if is_possible_24(nums):
                results.append(nums)
                valid_24_count += 1
                
                # 直接写入文件
                with open(output_file, "a") as f:
                    f.write(" ".join(map(str, nums)) + "\n")
                
                # 每生成1000个打印进度信息
                if valid_24_count % 1000 == 0:
                    print(f"已生成 {valid_24_count} 组可以构造24的数据")
            
            pbar.update(1)
    
    print(f"成功生成并保存了{valid_24_count}组可以构造出24的数字组合到{output_file}")
    return results

def get_prompts():
    """
    从文件中读取所有的数字组合，并组成一个prompt列表
    组成prompt的方法：
    1. 首先读取/home/weishaohang/workspace/24-Game-Reasoning/data/24_inputs.txt文件中每一行的4个数字，嵌入到/home/weishaohang/workspace/24-Game-Reasoning/templates/question.txt文件的4个数字的位置
    2. 嵌入后的字符串作为prompt，嵌入到/home/weishaohang/workspace/24-Game-Reasoning/templates/r1_prompt.txt中的{prompt}位置
    
    请你返回两个prompts列表，一个是24的prompt，一个是not_24的prompt
    """
    # 获取问题模板和r1 prompt模板
    with open("/home/weishaohang/workspace/24-Game-Reasoning/templates/question.txt", "r") as f:
        question_template = f.read()
    
    with open("/home/weishaohang/workspace/24-Game-Reasoning/templates/r1_prompt.txt", "r") as f:
        r1_template = f.read()
    
    # 读取可以构造24的数字组合
    valid_24_prompts = []
    with open("/home/weishaohang/workspace/24-Game-Reasoning/data/24_inputs.txt", "r") as f:
        for line in f:
            numbers = line.strip().split()
            if len(numbers) != 4:
                continue
            
            # 创建问题文本
            question = question_template.format(
                card1=numbers[0], 
                card2=numbers[1], 
                card3=numbers[2], 
                card4=numbers[3]
            )
            
            # 创建最终的prompt
            prompt = r1_template.format(prompt=question)
            valid_24_prompts.append(prompt)
    
    # 读取不能构造24的数字组合
    not_24_prompts = []
    with open("/home/weishaohang/workspace/24-Game-Reasoning/data/not_24_inputs.txt", "r") as f:
        for line in f:
            numbers = line.strip().split()
            if len(numbers) != 4:
                continue
            
            # 创建问题文本
            question = question_template.format(
                card1=numbers[0], 
                card2=numbers[1], 
                card3=numbers[2], 
                card4=numbers[3]
            )
            
            # 创建最终的prompt
            prompt = r1_template.format(prompt=question)
            not_24_prompts.append(prompt)
    
    return valid_24_prompts, not_24_prompts

def extract_think_answer(result):
    """
    从模型输出的结果中提取思考过程和答案
    
    :param result: 模型输出的原始结果字符串
    :return: 包含思考过程和答案的元组 (reasoning_content, answer)
    """
    reasoning_content = ""
    answer = ""
    
    # 提取<think>...</think>中的内容
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, result, re.DOTALL)
    if think_match:
        reasoning_content = think_match.group(1).strip()
    
    # 提取<answer>...</answer>中的内容
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, result, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    
    return reasoning_content, answer

def extract_reasoning_from_result(result):
    """
    从模型输出的完整结果中提取推理过程，移除答案部分
    
    :param result: 模型输出的原始结果字符串
    :return: 提取出的推理过程内容
    """
    # 查找并移除 <answer>...</answer> 部分
    answer_pattern = r'\n*<answer>.*?</answer>\n*'
    cleaned_result = re.sub(answer_pattern, '', result, flags=re.DOTALL)
    
    # 清理结尾可能的多余空白行
    cleaned_result = cleaned_result.rstrip()
    
    return cleaned_result

def process_existing_results():
    """
    处理已存在的结果文件，提取推理过程并更新结果
    """
    data_dir = os.path.join("/home/weishaohang/workspace/24-Game-Reasoning", "data")
    
    # 处理可构造24的结果
    valid_24_output_file = os.path.join(data_dir, "valid_24_results.json")
    if os.path.exists(valid_24_output_file):
        print("正在处理可构造24的结果...")
        with open(valid_24_output_file, "r", encoding="utf-8") as f:
            valid_24_processed_results = json.load(f)
        
        for i, result_dict in enumerate(valid_24_processed_results):
            if "result" in result_dict:
                reasoning_content = extract_reasoning_from_result(result_dict["result"])
                valid_24_processed_results[i]["reasoning_content"] = reasoning_content
        
        # 保存更新后的结果
        with open(valid_24_output_file, "w", encoding="utf-8") as f:
            json.dump(valid_24_processed_results, f, ensure_ascii=False, indent=4)
        
        print(f"成功更新可构造24的推理结果到 {valid_24_output_file}")
    else:
        print(f"文件不存在: {valid_24_output_file}")
    
    # 处理不可构造24的结果
    not_24_output_file = os.path.join(data_dir, "not_24_results.json")
    if os.path.exists(not_24_output_file):
        print("正在处理不可构造24的结果...")
        with open(not_24_output_file, "r", encoding="utf-8") as f:
            not_24_processed_results = json.load(f)
        
        for i, result_dict in enumerate(not_24_processed_results):
            if "result" in result_dict:
                reasoning_content = extract_reasoning_from_result(result_dict["result"])
                not_24_processed_results[i]["reasoning_content"] = reasoning_content
        
        # 保存更新后的结果
        with open(not_24_output_file, "w", encoding="utf-8") as f:
            json.dump(not_24_processed_results, f, ensure_ascii=False, indent=4)
        
        print(f"成功更新不可构造24的推理结果到 {not_24_output_file}")
    else:
        print(f"文件不存在: {not_24_output_file}")

def extract_question_from_prompt(prompt):
    """
    从prompt字符串中提取User和Assistant之间的内容作为问题
    
    :param prompt: 包含问题的完整prompt字符串
    :return: 提取出的问题内容
    """
    # 查找"User:"和"Assistant:"之间的内容
    user_pattern = r'User:(.*?)Assistant:'
    match = re.search(user_pattern, prompt, re.DOTALL)
    
    if match:
        question_content = match.group(1).strip()
        return question_content
    
    return ""

def add_question_to_results(results_file):
    """
    从JSON文件中读取结果，为每个结果字典添加question字段，然后保存回文件
    
    :param results_file: 结果JSON文件的路径
    """
    print(f"正在处理文件: {results_file}")
    
    # 读取JSON文件
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    # 为每个结果添加question字段
    for result_dict in results:
        if "prompt" in result_dict:
            question = extract_question_from_prompt(result_dict["prompt"])
            result_dict["question"] = question
    
    # 保存更新后的结果
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"成功为{len(results)}个结果添加question字段到 {results_file}")

def process_all_result_files():
    """
    处理所有结果文件，为每个结果添加问题字段
    """
    data_dir = os.path.join("/home/weishaohang/workspace/24-Game-Reasoning", "data")
    
    # 处理可构造24的结果
    valid_24_output_file = os.path.join(data_dir, "valid_24_results.json")
    if os.path.exists(valid_24_output_file):
        add_question_to_results(valid_24_output_file)
    else:
        print(f"文件不存在: {valid_24_output_file}")
    
    # 处理不可构造24的结果
    not_24_output_file = os.path.join(data_dir, "not_24_results.json")
    if os.path.exists(not_24_output_file):
        add_question_to_results(not_24_output_file)
    else:
        print(f"文件不存在: {not_24_output_file}")

def merge_result_files():
    """
    合并valid_24_results.json和not_24_results.json文件到一个新的JSON文件中
    
    :return: 合并后的结果列表
    """
    data_dir = os.path.join("/home/weishaohang/workspace/24-Game-Reasoning", "data")
    valid_24_output_file = os.path.join(data_dir, "valid_24_results.json")
    not_24_output_file = os.path.join(data_dir, "not_24_results.json")
    merged_output_file = os.path.join(data_dir, "all_24_game_results.json")
    
    merged_results = []
    
    # 读取可构造24的结果
    if os.path.exists(valid_24_output_file):
        print(f"读取文件: {valid_24_output_file}")
        try:
            with open(valid_24_output_file, "r", encoding="utf-8") as f:
                valid_24_results = json.load(f)
                
            # 为每个结果添加标签，表示它是可构造24的
            for result in valid_24_results:
                result["is_possible"] = True
                merged_results.append(result)
                
            print(f"成功添加了 {len(valid_24_results)} 个可构造24的结果")
        except Exception as e:
            print(f"读取 {valid_24_output_file} 时出错: {str(e)}")
    else:
        print(f"文件不存在: {valid_24_output_file}")
    
    # 读取不能构造24的结果
    if os.path.exists(not_24_output_file):
        print(f"读取文件: {not_24_output_file}")
        try:
            with open(not_24_output_file, "r", encoding="utf-8") as f:
                not_24_results = json.load(f)
                
            # 为每个结果添加标签，表示它是不可构造24的
            for result in not_24_results:
                result["is_possible"] = False
                merged_results.append(result)
                
            print(f"成功添加了 {len(not_24_results)} 个不可构造24的结果")
        except Exception as e:
            print(f"读取 {not_24_output_file} 时出错: {str(e)}")
    else:
        print(f"文件不存在: {not_24_output_file}")
    
    # 保存合并后的结果
    if merged_results:
        with open(merged_output_file, "w", encoding="utf-8") as f:
            json.dump(merged_results, f, ensure_ascii=False, indent=4)
        print(f"成功将 {len(merged_results)} 条结果合并并保存到 {merged_output_file}")
    else:
        print("没有结果可以合并")
    
    return merged_results

def shuffle_and_save_results(input_file=None, output_file=None):
    """
    对JSON文件中的结果进行随机打乱并保存
    
    :param input_file: 输入JSON文件路径，如果为None，则使用默认合并后的文件
    :param output_file: 输出JSON文件路径，如果为None，则在输入文件名基础上添加"_shuffled"
    :return: 打乱后的结果列表
    """
    data_dir = os.path.join("/home/weishaohang/workspace/24-Game-Reasoning", "data")
    
    # 设置默认输入输出文件
    if input_file is None:
        input_file = os.path.join(data_dir, "all_24_game_results.json")
    
    if output_file is None:
        # 从输入文件名生成输出文件名（添加_shuffled后缀）
        file_name, file_ext = os.path.splitext(os.path.basename(input_file))
        output_file = os.path.join(data_dir, f"{file_name}_shuffled{file_ext}")
    
    # 读取JSON文件
    if os.path.exists(input_file):
        print(f"读取文件: {input_file}")
        try:
            with open(input_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            
            # 记录原始数据长度
            original_count = len(results)
            print(f"读取到 {original_count} 条结果")
            
            # 随机打乱结果
            random.shuffle(results)
            
            # 保存打乱后的结果
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            
            print(f"成功将随机打乱后的 {len(results)} 条结果保存到 {output_file}")
            return results
        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {str(e)}")
            return []
    else:
        print(f"文件不存在: {input_file}")
        return []

def main():
    # 创建数据目录
    data_dir = os.path.join("/home/weishaohang/workspace/24-Game-Reasoning", "data")
    # model = ModelInference(model_type="deepseek")
    # if not os.path.exists(data_dir):
    #     os.makedirs(data_dir)
    
    # # 生成不能构造出24的数字组合
    not_24_file = os.path.join(data_dir, "not_24_inputs.txt")
    # gen_not_24_inputs(num_samples=300000, max_val=13, output_file=not_24_file)
    
    # # 生成可以构造出24的数字组合
    valid_24_file = os.path.join(data_dir, "24_inputs.txt")
    # gen_all_24_inputs(max_val=13, output_file=valid_24_file)
    
    # # 获取所有提示
    # valid_24_prompts, not_24_prompts = get_prompts()
    
    # # print(f"获取到 {len(valid_24_prompts)} 个可构造24的提示和 {len(not_24_prompts)} 个不可构造24的提示")
    

    
    # # # 对可构造24的提示进行推理
    # print("开始对可构造24的提示进行推理...")
    # valid_24_raw_results = model.inference(valid_24_prompts, asyn=True, is_r1=True)
    
    # # # 处理和保存结果
    # valid_24_processed_results = []
    
    # for i, result in enumerate(valid_24_raw_results):
    #     reasoning_content, answer = extract_think_answer(result)
        
    #     result_dict = {
    #         "prompt": valid_24_prompts[i],
    #         "reasoning_content": reasoning_content,
    #         "answer": answer,
    #         "result": result
    #     }
        
    #     valid_24_processed_results.append(result_dict)
    
    # # # 保存处理后的结果
    valid_24_output_file = os.path.join(data_dir, "valid_24_results.json")
    # with open(valid_24_output_file, "w", encoding="utf-8") as f:
    #     json.dump(valid_24_processed_results, f, ensure_ascii=False, indent=4)
    
    # print(f"成功保存可构造24的推理结果到 {valid_24_output_file}")
    
    # # 对不可构造24的提示进行推理
    # print("开始对不可构造24的提示进行推理...")
    # not_24_raw_results = model.inference(not_24_prompts, asyn=True, is_r1=True)
    
    # # 处理和保存结果
    # not_24_processed_results = []
    
    # for i, result in enumerate(not_24_raw_results):
    #     reasoning_content, answer = extract_think_answer(result)
        
    #     result_dict = {
    #         "prompt": not_24_prompts[i],
    #         "reasoning_content": reasoning_content,
    #         "answer": answer,
    #         "result": result
    #     }
        
    #     not_24_processed_results.append(result_dict)
    
    # # 保存处理后的结果
    not_24_output_file = os.path.join(data_dir, "not_24_results.json")
    # with open(not_24_output_file, "w", encoding="utf-8") as f:
    #     json.dump(not_24_processed_results, f, ensure_ascii=False, indent=4)
    
    # print(f"成功保存不可构造24的推理结果到 {not_24_output_file}")

    # # 处理已有的结果文件，提取推理过程
    # process_existing_results()

    # # 添加新的处理步骤：为结果添加问题字段
    # process_all_result_files()
    
    # 合并结果文件
    # merge_result_files()
    
    # 添加新功能：随机打乱合并后的结果并保存
    shuffle_and_save_results()


if __name__ == "__main__":
    main()