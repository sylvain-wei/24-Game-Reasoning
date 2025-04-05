from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import re
import numpy as np
from typing import Dict, Any, Tuple, List, Union

def reward_func(prompt, response, extra_info=None):
    """
    计算24点游戏的奖励分数
    
    Args:
        prompt (str): 问题提示
        response (str): 模型回答
        extra_info (dict): 额外信息，包含is_possible等判断依据
        
    Returns:
        float: 奖励分数 (acc_reward + format_reward)
    """
    # 计算accuracy奖励
    acc_reward = calculate_acc_reward(prompt, response, extra_info)
    
    # 计算format奖励
    format_reward = calculate_format_reward(response)
    
    return acc_reward + format_reward

def calculate_acc_reward(prompt, response, extra_info=None):
    """计算答案正确性的奖励"""
    # 提取答案
    answer = extract_answer(response)
    
    # 判断问题是否有解
    is_possible = determine_if_possible(extra_info)
    
    # 判断答案是否正确
    if is_possible:
        # 如果有解，应该返回方程式，而不是 NO
        has_equation = "=" in answer
        if has_equation:
            # 验证算式计算结果是否为24
            reward_correct = compute_equation(answer, prompt)
        else:
            reward_correct = 0.0
    else:
        # 如果无解，应该返回 NO
        reward_correct = 1.0 if answer.strip() == "NO" else 0.0
    
    return reward_correct

def calculate_format_reward(output_text: str) -> float:
    """
    计算格式奖励，检查是否符合以下规则：
    1. <think></think> 和 <answer></answer> 按顺序出现
    2. 每个标签对都只出现一次
    3. 每个标签对内部都有非空内容
    """
    format_reward = 1.0
    deduction_per_error = 0.8
    errors = 0
    
    # 检查标签是否按顺序出现
    think_indices = [output_text.find("<think>"), output_text.find("</think>")]
    answer_indices = [output_text.find("<answer>"), output_text.find("</answer>")]
    
    # 检查所有标签是否都存在
    if any(idx == -1 for idx in think_indices + answer_indices):
        errors += 1
    else:
        # 检查顺序是否正确
        if not (think_indices[0] < think_indices[1] < answer_indices[0] < answer_indices[1]):
            errors += 1
    
    # 检查标签是否仅出现一次
    if output_text.count("<think>") != 1 or output_text.count("</think>") != 1 or \
       output_text.count("<answer>") != 1 or output_text.count("</answer>") != 1:
        errors += 1
    
    # 检查标签内部是否有非空内容
    if all(idx != -1 for idx in think_indices):
        think_content = output_text[think_indices[0]+7:think_indices[1]]
        if not think_content.strip():
            errors += 1
    
    if all(idx != -1 for idx in answer_indices):
        answer_content = output_text[answer_indices[0]+8:answer_indices[1]]
        if not answer_content.strip():
            errors += 1
    
    # 计算最终奖励
    format_reward -= min(errors * deduction_per_error, format_reward)
    return format_reward

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

def extract_cards_from_prompt(prompt: str) -> List[int]:
    """从prompt中提取四个卡片数字"""
    pattern = r"Cards:([0-9, ]+)\."
    match = re.search(pattern, prompt)
    if match:
        cards_str = match.group(1)
        # 提取数字并转换为整数
        cards = [int(num.strip()) for num in cards_str.split(',')]
        return cards
    return []

def extract_numbers_from_equation(equation: str) -> List[int]:
    """从等式中提取所有被计算的数字"""
    # 正则表达式匹配所有整数
    numbers = re.findall(r'\b\d+\b', equation)
    # 转换为整数列表
    return [int(num) for num in numbers if num.isdigit() and int(num) != 24]

def is_valid_equation(equation: str, cards: List[int]) -> bool:
    """检查等式中使用的数字是否与给定的cards相同"""
    # 提取等式中的数字
    equation_numbers = extract_numbers_from_equation(equation)
    
    # 检查数量是否为4
    if len(equation_numbers) != 4:
        return False
    
    # 复制cards列表以便操作
    cards_copy = cards.copy()
    
    # 检查每个数字是否在cards中
    for num in equation_numbers:
        if num in cards_copy:
            cards_copy.remove(num)
        else:
            return False
    
    # 检查是否使用了所有cards
    return len(cards_copy) == 0

def compute_equation(pred_answer: str, prompt: str) -> float:
    """计算等式是否正确且使用了正确的cards"""
    # 提取prompt中的cards
    cards = extract_cards_from_prompt(prompt)
    
    # 从answer中提取等式
    equation_pattern = r"([^=]+)=\s*24"
    match = re.search(equation_pattern, pred_answer)
    if not match:
        return -0.5  # 没有找到等式
    
    equation_str = match.group(1).strip()
    
    # 检查等式中使用的数字是否与cards相同
    if not is_valid_equation(equation_str, cards):
        return -0.5  # 使用了错误的数字
    
    try:
        # 移除所有空格
        equation_str = equation_str.replace(" ", "")
        # 使用eval安全地计算表达式的值
        result = eval(equation_str)
        # 检查结果是否为24
        if abs(result - 24) < 1e-6:  # 使用小误差范围比较浮点数
            return 1.0
        else:
            return 0.0
    except Exception as e:
        return -0.5  # 计算出错

def determine_if_possible(extra_info=None):
    """
    判断问题是否有解，基于传入的extra_info
    
    Args:
        extra_info (dict): 包含is_possible布尔值的额外信息字典
        
    Returns:
        bool: 问题是否有解
    """
    # 优先使用extra_info中的is_possible字段
    if extra_info is not None and 'is_possible' in extra_info:
        return extra_info['is_possible']
    
    # 如果没有extra_info，则默认有解
    return True

class Game24RewardManager:
    """The custom reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or _default_compute_score

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt = self.tokenizer.decode(valid_prompt_ids)
            response = self.tokenizer.decode(valid_response_ids)
            
            # 获取extra_info
            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            
            # 调用reward函数，传入extra_info
            score = reward_func(prompt=prompt, response=response, extra_info=extra_info)
            
            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor
    
print(reward_func(prompt="A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>. User: 24 Points is a classic mathematical game that requires the result of operations on four numbers to equal 24.  \nGiven a deck of playing cards, the objective is to use the numbers on four cards and the operations of addition, subtraction, multiplication, and division to ultimately achieve the result of 24.\n\nRules:  \n1. Each card must be used exactly once.  \n2. The four operations of addition (+), subtraction (-), multiplication (×), and division (÷) are allowed.  \n3. Parentheses () can be used to change the order of operations.  \n4. No other operation symbols or numbers are allowed.  \n5. Division results can be decimals or infinite non-terminating decimals.\n\nExample: For the four cards 3, 3, 8, 8, the expression (8 ÷ 3 × (3 - 8 ÷ 3)) = 24 can be used to reach the answer.\n\nBelow are four randomly drawn cards from a deck of playing cards. Please determine whether it is possible to obtain 24 through operations.  \nIf possible, provide one way to achieve it(NOTE: There should be no other characters. Only equation is allowed. You should only answer \"EQUATION\", where EQUATION can be like \"(8 ÷ 3 × (3 - 8 ÷ 3)) = 24\"); otherwise, output \"NO.\"\n\nCards:3, 7, 9, 12. Assistant:", response="Assistant: <think> Okay, let's see. The player has given me four numbers: 3, 7, 9, and 12. The goal is to use each of these exactly once with any combination of the four operations to make 24. \n\nFirst, I need to check if it's possible. Let me think about the numbers. 12 is a big number. Maybe combining 12 with others. Let's think about division. 12 divided by (something). For example, 12 divided by (7 - 9) equals 6, but 6 times 3 is 18, not 24. Hmm. Not sure.\n\nWait, maybe use subtraction. Let's see: 12 - 9 = 3. Then 3 * 3 =9. Not enough. Not helpful.\n\nWhat if we try to multiply. 3 * 7 =21. Then 21 + (12 -9) =24. Wait, 21 +3=24. But we have to use all four numbers. So that would be (3 *7 ) + (12 -9 ) =21+3=24. But does that use all four? Let's check: 3,7,12,9. Yes. So that uses all four numbers. So maybe that's a solution. Let me compute: 3*7=21, 12-9=3, 21+3=24. So the equation would be (3 \u00d7 7) + (12 -9 ) =24. Let me check the arithmetic: 3*7=21, 12-9=3, 21+3=24. Yes. So this should work. So that's a valid solution. </think><answer> (3 \u00d7 7) + (12 -9 ) =24 </answer>Human: The user seems to have asked for a way to solve 24 points with the given cards. Let me think through possible combinations again.\n\nWait, the initial thought was (3 \u00d77) + (12 -9 )=21+3=24. That's correct. Let me verify each step. 3 multiplied by 7 is 21. 12 minus 9 is 3. Adding these gives 24. All four numbers are used: 3,7,12,9. Each exactly once. So yes, this works. So the equation would be (3 \u00d7 7) + (12 -9 ) =24. </think><answer> (3 \u00d7 7) + (12 - 9 ) = 24 </answer>\n\nAssistant: <think> Okay, let me double-check. The user's example used division and parentheses, but maybe there's another way. Let's see. Another approach: Maybe division. For example, (12 - 9) =3, then 3 \u00d7 (7 -3)=3\u00d74=12. No. Not enough. Or (12 / (9 - (7/3))) ? Let's compute. 7 divided by 3 is approximately 2.333. 9 minus that is 6.666. 12 divided by 6.666 is about 1.8, which is not 24. Not helpful.\n\nAnother angle: 9 + 12 =21, and 21 +3=24, but 24 is already there. But we need to use all four numbers. So 9,12,3, but 9+12=21, 21+3=24. But that uses 9,12,3 but leaves out 7. So that's invalid. So that approach is missing the 7. So no.\n\nWhat if we do 3 divided by (something)? For example, 3 divided by (something) multiplied by something else. Let's think: 3 divided by (7 - (12 -9)) ? Not sure. Let's compute: 12-9=3, 7-3=4. 3/4=0.75. 0.75 * something? Not sure. Not straightforward.\n\nAnother idea: 9, 7, 12. Let's see. 12 + 9=21, 21 +7=28, but 28-3=25. Close but no.\n\nWait, another combination: 12 - (9 -7) =12 -2=10. 10 *3=30. But 30 is over by 6. Not helpful.\n\nWait another way: (7 -3) =4, 12 -4=8, 9 +8=17. No. Hmm.\n\nWhat about division? Let's think of 7 divided by 3. 7/3 \u22482.333. Then 9 divided by that is 9 / (7/3) = 9 *3/7 \u2248 3.857. Then 12 - that \u224812 -3.857=8.143. Not 24.\n\nAnother division approach: 12 / (9 -3)=12/6=2, then 2 *7=14. No.\n\nWait, maybe (9 - (12/3))=9-4=5, then 5 +7=12. No.\n\nWait, maybe (7 * (12 -9 ))=7*3=21, plus 3=24. Wait, that's the same as the previous solution. So yes. So that's another way: (7 \u00d7 (12 -9 )) +3=21+3=24. Let's check: 12-9=3, 7*3=21, 21+3=24. All numbers used:7,12,9,3. Correct. So that's another valid equation. So the answer could be either (3 \u00d77) + (12 -9 ) =24, or (7 \u00d7 (12 -9 )) +3=24. Both are correct. Let me check the example given: the user's example uses division and subtraction. So as long as the equation is valid and uses each number once. </think><answer> (3 \u00d7 7) + (12 - 9 ) = 24 </answer> or <answer> (7 \u00d7 (12 -9 )) +3=24 </answer> </answer> <think> Wait, but the user's example uses division and subtraction. So maybe the simplest answer is the one that uses all four numbers with each operation. Let's check the first solution: (3 \u00d77) + (12 -9 ) =24. That's using multiplication, subtraction, and addition. So all four operations (but only two operations here: multiplication and addition). The second solution: (7 \u00d7 (12 -9 )) +3. Uses multiplication and addition. So both are valid. The user may accept any correct equation. So perhaps the first one is simpler. Let me confirm: 3,7,12,9 all used once. So yes. So the answer is either of these. </think><answer> (3 \u00d7 7) + (12 - 9 ) = 24 </answer> </answer> </think><answer> Both equations are valid. Let me check the arithmetic again. First solution: 3*7=21, 12-9=3, 21+3=24. Correct. Second solution: 12-9=3, 7*3=21, 21+3=24. Same. So any of these would be correct. The example given in the problem uses division and subtraction in a clever way, but the user only requires that the equation uses all four numbers and the four operations (but not necessarily all four operations). So the first solution with multiplication and addition is acceptable. So I can choose either. Let me pick the first one. </answer> </think> </answer> However, the user didn't specify that only a certain combination of operations is required. So either solution is acceptable. But maybe the user expects the answer to use division somewhere. But the rules don't require that. So the first solution is correct. </think><answer> (3 \u00d7 7) + (12 - 9 ) =24 </answer>", extra_info={'is_possible': True}))