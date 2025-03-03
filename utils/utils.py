import time
import requests
from typing import Union, List
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

class ModelInference:
    def __init__(self, model_type: str = None, openai_api_key: str = None, model_path: str = None):
        """
        初始化模型推理类，选择是否使用GPT-4 API或开源模型。
        
        :param use_gpt: 是否使用GPT-4 API（默认为False，使用开源模型）
        :param openai_api_key: GPT-4 API的密钥（仅在use_gpt为True时需要提供）
        :param model_path: 开源模型的路径（仅在use_gpt为False时需要提供）
        """
        self.model_type = model_type
        
        if 'gpt' in model_type.lower():
            if openai_api_key is None:
                raise ValueError("OpenAI API key is required for GPT inference")
            self.openai_api_key = openai_api_key
        elif 'vllm' in model_type.lower():
            if model_path is None:
                raise ValueError("Model path is required for VLLM inference")
            self.model_path = model_path
            self.model = LLM(
                model=model_path,
                gpu_memory_utilization=1,
                device="cuda", # 不设置也行
                dtype="bfloat16", # 不设置也行
                tensor_parallel_size=2, # 4卡
                enforce_eager=True, # 不设置也行
                enable_chunked_prefill=True, # 最好设置，否则可能会爆显存
                max_model_len=76700,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.sampling_params = SamplingParams(
                n=1,
                temperature=0.1,
                max_tokens=128000,
                stop=[self.tokenizer.eos_token]
            )
        elif 'deepseek' in model_type.lower():
            pass

    def call_gpt(self, prompt: str, asyn: bool=True, max_workers=50) -> str:
        """
        通过HTTP请求调用OpenAI GPT-4 API进行推理
        
        :param prompt: 输入的文本提示
        :return: GPT-4的响应文本
        """
        # client = OpenAI(api_key="YOUR_KEY", base_url="https://api.siliconflow.cn/v1")
        # def _call_gpt_api(prompt):
        #     response = client.chat.completions.create(
        #         model="deepseek-ai/DeepSeek-V3" if not is_r1 else "deepseek-ai/DeepSeek-R1",  # 使用的模型
        #         messages=[
        #             {"role": "system", "content": "You are a helpful assistant"},
        #             {"role": "user", "content": prompt},
        #         ],
        #         stream=False,
        #         temperature=1.0,
        #     )
        #     return response.choices[0].message.content
        
        # if asyn:
        #     messages = prompts if isinstance(prompts, list) else [prompts]
            
        #     with ThreadPoolExecutor(max_workers=max_workers) as executor:
        #         # 提交任务到线程池
        #         ret = executor.map(_call_deepseek_api, messages)
        #         results = [res for res in ret]
        #         return results
        # else:
        #     return _call_deepseek_api(prompts if isinstance(prompts, str) else prompts[0])
        # url = "https://api.openai.com/v1/completions"
        # headers = {
        #     "Authorization": f"Bearer {self.openai_api_key}",
        #     "Content-Type": "application/json"
        # }
        # data = {
        #     "model": "gpt-4",
        #     "prompt": prompt,
        #     "max_tokens": 150
        # }
        
        # 发送请求
        # response = requests.post(url, headers=headers, json=data)
        
        # if response.status_code == 200:
        #     return response.json()['choices'][0]['text'].strip()
        # else:
        #     raise Exception(f"Error calling GPT API: {response.status_code} - {response.text}")
        pass
    
    def call_vllm(self, prompts) -> str:
        """
        使用VLLM框架推理开源模型（如Llama-3.1-8B-Instruct）。
        
        :param prompt: 输入的文本提示
        :return: 模型的响应文本
        """
        if 'instruct' in self.model_path.lower():
            prompts = [
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                for prompt in prompts
            ]
        
        responses = self.model.generate(prompts, sampling_params=self.sampling_params)
        # print(prompts)
        # print(response)
        outputs = []
        for response in responses:
            for sub_output in response.outputs:
                outputs.append(sub_output.text)
        return outputs
    
    def call_deepseek(self, prompts:Union[str, List[str]], asyn=True, max_workers=100, is_r1=False)->str:
        # TODO: 记得匿名化api_key
        # 初始化 OpenAI 客户端
        # client = OpenAI(api_key="YOUR_KEY", base_url="https://api.deepseek.com")
        # 定义一个函数来调用 API
        # def _call_deepseek_api(prompt):
        #     response = client.chat.completions.create(
        #         model="deepseek-chat" if not is_r1 else "deepseek-reasoner",  # 使用的模型
        #         messages=[
        #             {"role": "system", "content": "You are a helpful assistant"},
        #             {"role": "user", "content": prompt},
        #         ],
        #         stream=False,
        #         temperature=1.0,
        #     )
        #     return response.choices[0].message.content
        
        # @backoff.on_exception(backoff.expo, openai.RateLimitError)
        
        # NOTE:硅基流动
        # client = OpenAI(api_key="YOUR_KEY", base_url="https://api.siliconflow.cn/v1")
        # NOTE 火山引擎
        client = OpenAI(api_key="YOUR_KEY", base_url="https://ark.cn-beijing.volces.com/api/v3")
        def _call_deepseek_api(prompt):
            # Calculate the delay based on your rate limit
            rate_limit_per_minute = 30000
            delay = 60.0 / rate_limit_per_minute
            # Sleep for the delay
            time.sleep(delay)

            # 重试机制
            retries = 10000000  # 最大重试次数
            for attempt in range(retries):
                try:
                    response = client.chat.completions.create(
                        model="ep-20250223144841-bllbh" if not is_r1 else "ep-20250218232341-bht2q",  # 使用的模型 | deepseek-ai/DeepSeek-R1
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": prompt},
                        ],
                        stream=False,
                        temperature=1.0 if not is_r1 else 0.6,
                    )
                    return response.choices[0].message.content  if not is_r1 else response.choices[0].message.reasoning_content+response.choices[0].message.content
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == retries - 1:
                        raise e  # 如果是最后一次失败，则抛出异常
                    time.sleep(1)  # 重试前等待 1 秒
        # def _call_deepseek_api(prompt):
        #     # Calculate the delay based on your rate limit
        #     rate_limit_per_minute = 100
        #     delay = 60.0 / rate_limit_per_minute
        #     # Sleep for the delay
        #     time.sleep(delay)
        #     response = client.chat.completions.create(
        #         model="deepseek-ai/DeepSeek-V3" if not is_r1 else "deepseek-ai/DeepSeek-R1",  # 使用的模型
        #         messages=[
        #             {"role": "system", "content": "You are a helpful assistant"},
        #             {"role": "user", "content": prompt},
        #         ],
        #         stream=False,
        #         temperature=1.0,
        #     )
        #     return response.choices[0].message.content
        
        if asyn:
            messages = prompts if isinstance(prompts, list) else [prompts]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交任务到线程池
                ret = executor.map(_call_deepseek_api, messages)
                results = [res for res in ret]
                return results
        else:
            return _call_deepseek_api(prompts if isinstance(prompts, str) else prompts[0])
        
    def inference(self, prompts, asyn=True, is_r1=False) -> str:
        """
        根据初始化时的配置，选择调用GPT-4 API或开源模型推理
        
        :param prompt: 输入的文本提示
        :return: 模型的响应文本
        """
        if 'gpt' in self.model_type.lower():
            return self.call_gpt(prompts)
        elif 'vllm' in self.model_type.lower():
            return self.call_vllm(prompts)
        elif 'deepseek' in self.model_type.lower():
            return self.call_deepseek(prompts, asyn=asyn, is_r1=is_r1)
        else:
            raise ValueError("Invalid model type")