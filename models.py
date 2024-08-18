import requests
from http import HTTPStatus
from dashscope import Generation
import dashscope
from dashscope.api_entities.dashscope_response import Role
import os
import re
import json
import random
import base64

from tqdm import tqdm
import numpy as np

from openai import OpenAI, BadRequestError

import transformers
import torch
from huggingface_hub import login
import deepspeed
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from typing import List 
import anthropic
import erniebot

erniebot_api_type = 'aistudio'
erniebot_api_key = ''
gemini_api_key = ''
openai_api_key = ''
claude_api_key=''


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, token_id_list: List[int] = None):
        self.token_id_list = token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list

 
class GPT4:

    def __init__(self):
        # Get OpenAI API Key from environment variable
        api_key = openai_api_key
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt, temperature=0.0):
        prompt = "You are required to solve a programming problem with python. Please enclose your code inside a ```python``` block. " \
                + prompt

        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content,prompt


    def extract_code(self, response):
        pattern = r"```python(.*?)```"

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0]
        else:
            return response
            
class GPT3_5:

    def __init__(self):
        # Get OpenAI API Key from environment variable
        api_key = openai_api_key
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt, temperature=0.0):
        prompt = "You are required to solve a programming problem with python. Please enclose your code inside a ```python``` block. " \
                + prompt

        
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content,prompt


    def extract_code(self, response):
        pattern = r"```python(.*?)```"

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0]
        else:
            return response
             
class Claude3:

    def __init__(self):
        # Get OpenAI API Key from environment variable
        api_key = claude_api_key
        self.client = anthropic.Anthropic( api_key = api_key)

    def generate(self, prompt, temperature=0.0):
        prompt = "You are required to solve a programming problem with python. Please enclose your code inside a ```python``` block. " \
                + prompt
 

        message = self.client.messages.create(
            model="claude-3-opus-20240229", # 模型型號
            max_tokens=2048, # 選用，回傳token的最大長度，避免爆預算
            temperature=temperature,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return message.content[0].text,prompt
 

    def extract_code(self, response):
        pattern = r"```python(.*?)```"

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0]
        else:
            return response
        

class GEMINI_PRO:
    def __init__(self):
        import google.generativeai as genai
        # Get Google API Key from environment variable
#         api_key = os.environ["GOOGLE_API_KEY"]
        api_key = gemini_api_key
        genai.configure(api_key=api_key)
        self.genai_package = genai
        self.client = genai.GenerativeModel('gemini-pro')

    def generate(self, prompt, temperature=0.0):
        # prompt = "You are required to solve a programming problem. Please enclose your code inside a ```python``` block. " \
        #     "Do not write a main() function. If Call-Based format is used, return the result in an appropriate place instead of printing it.\n\n" \
        #          + prompt
        prompt += "\nANSWER:\n"
        prompt = "You are required to solve a programming problem with python. Please enclose your code inside a ```python``` block. " \
                 + prompt

        try:
            response = self.client.generate_content(prompt,
                generation_config=self.genai_package.types.GenerationConfig(
                    temperature=temperature
                )
            )
            return response.text,prompt
        except Exception as e:
            print(e)
            return None

    def extract_code(self, response):
        pattern = r"```python(.*?)```"

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0]
        else:
            return response

class Wenxin: 
    def __init__(self):
        erniebot.api_type = erniebot_api_type
        erniebot.access_token = erniebot_api_key

    def generate(self, prompt, temperature=0.0):
        
        prompt += "\nANSWER:\n"
        prompt = "You are required to solve a programming problem with python. Please enclose your code inside a ```python``` block. " \
                 + prompt

        try:
            response = erniebot.ChatCompletion.create(
                model='ernie-3.5',
                messages=[{
                    'role': 'user',
                    'content': prompt
                }])
            
            return response.get_result(),prompt

        except Exception as e:
            print(e)
            return None

    def extract_code(self, response):
        pattern = r"```python(.*?)```"

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0]
        else:
            return response

class Qwen():
    def __init__(self):
        dashscope.api_key = '' # 将 YOUR_API_KEY 改成您创建的 API-KEY
        
    

    def generate(self, prompt, temperature=0.0):
     
        prompt += "\nANSWER:\n"
        prompt = "You are required to solve a programming problem with python. Please enclose your code inside a ```python``` block. " \
                 + prompt
        
        messages = [{'role': 'user', 'content': prompt}]
        gen = Generation()
        
        try:
            response = gen.call(
            Generation.Models.qwen_turbo,
            messages=messages,
            temperature = temperature,
            result_format='message', # 设置结果为消息格式
            )
            if response ==None:
                response = ''
            else:
                response = response.output.choices[0]['message']['content']
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response):
        pattern = r"```python(.*?)```"

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[-1]
        else:
            return response


class ChatGLM():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/chatglm3-6b",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "THUDM/chatglm3-6b",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            token=None,
            cache_dir=None
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, prompt, temperature=0.0):
     
        prompt += "\nANSWER:\n"
        prompt = "You are required to solve a programming problem with python. Please enclose your code inside a ```python``` block. " \
                 + prompt
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response):
        pattern = r"```python(.*?)```"

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[-1]
        else:
            return response

class Llama2():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.access_token = '' 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        login(self.access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto', 
            token=None,
            cache_dir=None
        ).to(self.device)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    def generate(self, prompt, temperature=0.0):
     
        prompt += "\nANSWER:\n"
        prompt = f"Please provide the python code based on the question and enclose your code inside a ```python``` block.\n{prompt}"
        prompt = f"<s>[INST] {prompt} [/INST] "
        # <s>[INST] Using this information : {context} answer the Question : {query} [/INST] 
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # inputs = self.tokenizer.apply_chat_template(prompt, return_tensors="pt").to(self.device)
        try:
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True, 
                pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response):
        pattern = r'```([\s\S]*?)```'
        matches = re.findall(pattern, response) 
    
        if matches:
            return matches[-1]
        else:
            return response

class StarCoder2():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.access_token = "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"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        login(self.access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "bigcode/starcoder2-7b",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "bigcode/starcoder2-7b",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            token=None,
            cache_dir=None
        ).to(self.device)
        
    
 
    def generate(self, prompt, temperature=0.0):
     
        prompt += "\nANSWER:\n"
        prompt = f"Please provide the python code based on the question.\n{prompt}"
         
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # inputs = self.tokenizer.apply_chat_template(prompt, return_tensors="pt").to(self.device)
        try:
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response):
        response = response.split("ANSWER:", maxsplit=1)[-1].strip()
        return response

class Mixtral_7B():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.access_token = "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"
        
        login(self.access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-Instruct-v0.2",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            token=None,
            cache_dir=None
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    def generate(self, prompt, temperature=0.0):
     
        prompt += "\nANSWER:\n"
        prompt = f"Please provide the python code based on the question and enclose your code inside a ```python``` block.\n{prompt}"
        prompt = f"<s>[INST] {prompt} [/INST] "
        # <s>[INST] Using this information : {context} answer the Question : {query} [/INST] 
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # inputs = self.tokenizer.apply_chat_template(prompt, return_tensors="pt").to(self.device)
        try:
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response):
        pattern = r"```python(.*?)```"

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[-1]
        else:
            return response

class Mixtral_8x_7B():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.access_token = "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"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        login(self.access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            token=None,
            cache_dir=None
        ).to(self.device)
        
        
 
    def generate(self, prompt, temperature=0.0):
     
        prompt += "\nANSWER:\n"
        prompt = f"Please provide the python code based on the question and enclose your code inside a ```python``` block.\n{prompt}"
        prompt = f"<s>[INST] {prompt} [/INST] "
        # <s>[INST] Using this information : {context} answer the Question : {query} [/INST] 
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        # inputs = self.tokenizer.apply_chat_template(prompt, return_tensors="pt").to(self.device)
        try:
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response):
        pattern = r"```python(.*?)```"

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[-1]
        else:
            return response

class Gemma():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.access_token = "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"
        
        login(self.access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "google/gemma-7b",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-7b",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            token=None,
            cache_dir=None
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    def generate(self, prompt, temperature=0.0):
     
        prompt += "\nANSWER:\n"
        prompt = "You are required to solve a programming problem with python." \
                 + prompt
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response):
        # response = response.split("ANSWER:", maxsplit=1)[-1].strip()
        # return response
        pattern = r'ANSWER:(.*?)<eos>'
        match = re.search(pattern, response, re.DOTALL)
        
        if match:
            content = match.group(1).strip()
            return content
        else:
            return response

class CodeGeeX():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "THUDM/codegeex2-6b",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "THUDM/codegeex2-6b",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            token=None,
            cache_dir=None
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, prompt, temperature=0.0):
        
        prompt += "\nANSWER:\n"
        prompt = "# language: Python\n# " + prompt 
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response):
        # pattern = r"```python(.*?)```"

        response = response.split("ANSWER:", maxsplit=1)[-1].strip()
        return response
        
       
            
class CodeLLaMA_7b():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            "codellama/CodeLlama-7b-Instruct-hf",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "codellama/CodeLlama-7b-Instruct-hf",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            token=None,
            cache_dir=None
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, prompt, temperature=0.0):
     
        prompt = f'''As an expert code developer with years of experience, please provide the python code based on the question.\n{prompt}'''
        prompt = f'<s>[INST] {prompt} [/INST]'
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response):
        pattern = r'```([\s\S]*?)```'

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.search(pattern, response)

        if matches:
            return matches.group(1).strip()
        else:
            return response

class DeepSeekCoder():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            token=None,
            cache_dir=None
        )
        
        
    
    def generate(self, prompt, temperature=0.0):
        prompt = f"Please provide the python code based on the question and enclose your code inside a ```python``` block.\n{prompt}"
        prompt = f'''You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you only answer questions related to computer science. For politically sensitive questions, security and privacy issues, and other non-computer science questions, you will refuse to answer.
### Instruction:
{prompt}
### Response:
'''
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response):
        matches = re.search(r'### Response:(.*?)```python(.*?)```', response, re.DOTALL)
        
        if matches:
            return matches.group(2).strip()
        else:
            return response


class WizardCoder():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "WizardLM/WizardCoder-Python-7B-V1.0",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "WizardLM/WizardCoder-Python-7B-V1.0",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            token=None,
            cache_dir=None
        )
        
        

    def generate(self, prompt, temperature=0.0):
        prompt = f'''As an expert code developer with years of experience, please provide the python code based on the question and enclose your code inside a ```python``` block.
{prompt}'''
        
        prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{prompt}

### Response:"""
        
        
    
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response):
        matches = re.findall(r'### Response:(.*?)</s>', response, re.DOTALL)

        if matches:
            first_response = matches[-1].strip()
            pattern = r"```python(.*?)```"
            matches = re.findall(pattern, first_response, re.DOTALL)
            if matches:
                return matches[-1].strip()
            else:
                return first_response

        else:
            return response



class StarCoder():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.access_token = "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"
        
        login(self.access_token)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "HuggingFaceH4/starchat-beta",
            use_fast=True,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            "HuggingFaceH4/starchat-beta",
            torch_dtype=torch.float16,
            # load_in_4bit=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
        )
         
    

    def generate(self, prompt, temperature=0.0):
        prompt = f'''As an expert code developer with years of experience, please provide the python code based on the question and enclose your code inside a ```python``` block.
{prompt}'''
        system_message = 'Below is a dialogue between a human and an AI assistant called StarChat.'
        prompt = f'<|system|>\n{system_message.strip()}<|end|>\n<|user|>\n{prompt.strip()}<|end|>\n<|assistant|>'
        
        inputs = self.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(self.device)
        
        try:
            outputs = self.model.generate( 
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                # References: https://github.com/bigcode-project/starcoder/issues/73
                stopping_criteria=StoppingCriteriaList([
                    StopAtSpecificTokenCriteria(token_id_list=[
                        self.tokenizer.encode("<|end|>", return_tensors='pt').tolist()[0][0]
                    ])
                ])
            ).to('cpu') 
            output = outputs[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None
 
        
    def extract_code(self, response):
        pattern = r"```python(.*?)```"

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[-1]
        else:
            return response


class MagicCoder():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "ise-uiuc/Magicoder-S-DS-6.7B",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "ise-uiuc/Magicoder-S-DS-6.7B",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            token=None,
            cache_dir=None
        )
        
        

    def generate(self, prompt, temperature=0.0):
        prompt = f'''Please provide the python code based on the question and enclose your code inside a ```python``` block.
{prompt}'''
        prompt = f"""You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{prompt}

@@ Response
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            output = self.model.generate(
                inputs['input_ids'],
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            output = output[0].to("cpu")
            response = self.tokenizer.decode(output)
            return response,prompt
        except Exception as e:
            print(e)
            return None

        
    def extract_code(self, response): 
        matches = re.search(r'@@ Response(.*?)```python(.*?)```', response, re.DOTALL)
        
        if matches:
            return matches.group(2).strip()
        else:
            return response

class Llama3():
    def __init__(self):
        import transformers
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        self.access_token = '' 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        login(self.access_token)
   
        self.tokenizer = AutoTokenizer.from_pretrained(
           "meta-llama/Meta-Llama-3-8B-Instruct",
            use_fast=True,
            trust_remote_code=True,
            token=None,
            cache_dir=None
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map='auto',
            token=None,
            cache_dir=None
        )
        
        

    def generate(self, prompt, temperature=0.0):
        prompt = f'''Please provide the python code based on the question and enclose your code inside a ```python``` block.
{prompt}'''
        prompt += "\nANSWER:\n"
        messages = [
            {"role": "system", "content": "You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=2048,
                temperature=temperature,
                top_k=50,
                top_p=0.95,
                num_return_sequences=1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            ) 
        # generated_ids = self.model.generate(
        #     model_inputs.input_ids, 
        #     max_new_tokens=2048,
        #     pad_token_id=self.tokenizer.eos_token_id
        # )  
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        return response,prompt   
        


        
    def extract_code(self, response): 
        pattern = r"```python(.*?)```"

        # response = response.split("###ANSWER:", maxsplit=1)[-1].strip()
        
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[-1]
        else:
            return response
