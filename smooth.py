#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers import TrainingArguments
from safetensors.torch import save_model, load_model

from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm_qwen
from smoothquant.fake_quant import W8A8Linear, quantize_qwen
from smoothquant.calibration import get_act_scales

from datasets import config
#from vllm import LLM
from smoothquant.calibration import get_static_decoder_layer_scales

config.HF_DATASETS_CACHE = '/root/autodl-tmp/qwen/Qwen-7B'


class Evaluator:
    def __init__(self, dataset, tokenizer, device):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples["text"])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type="torch", columns=["input_ids"])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.dataset:
            input_ids = batch["input_ids"].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
        acc = hit / total
        return acc

model = AutoModelForCausalLM.from_pretrained(#'qwen/Qwen-7B',
  '/root/autodl-tmp/qwen/Qwen-7B',
  #cache_dir='/root/autodl-tmp/qwen/Qwen-7B',
  device_map='auto',
  torch_dtype=torch.float32,
  #load_in_8bit=True,
  trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('/root/autodl-tmp/qwen/Qwen-7B',trust_remote_code=True)
def print_model_size(model):
    # https://discuss.pytorch.org/t/finding-model-size/130275
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Model size: {:.3f}MB'.format(size_all_mb))
print_model_size(model)
model.get_memory_footprint()


# In[2]:


model=model.to(torch.float32)
model.get_memory_footprint()


# In[2]:


# from datasets import load_dataset
# #tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-30b')
# dataset = load_dataset('lambada', split='validation[:1000]')
# evaluator = Evaluator(dataset, tokenizer)
# acc_fp16, lantecy_fp16 = evaluator.evaluate(model)
# print(f'FP16 accuracy: {acc_fp16}, per-sample lantecy: {lantecy_fp16:.3f}ms')


# In[3]:


act_scales = torch.load("./qwen.pt")


# In[4]:


smooth_lm_qwen(model, act_scales, 0.5)


# In[5]:


model = quantize_qwen(model, act_scales)


# In[6]:


print_model_size(model)
model.get_memory_footprint()


# In[ ]:





# In[ ]:


#!export HF_ENDPOINT=https://hf-mirror.com
from optimum_benchmark.backends.pytorch.config import PyTorchConfig
from optimum_benchmark.benchmarks.inference.config import InferenceConfig
from optimum_benchmark.experiment import ExperimentConfig, launch
from optimum_benchmark.launchers.process.config import ProcessConfig
from optimum_benchmark.logging_utils import setup_logging
from optimum_benchmark.backends.pytorch.backend import PyTorchBackend
from optimum_benchmark.benchmarks.inference.benchmark import InferenceBenchmark
from optimum_benchmark.launchers.process.config import ProcessConfig
from transformers import AutoModelForCausalLM,AutoTokenizer

import json
from datasets import config

from smoothquant.smooth import smooth_lm_qwen
from smoothquant.fake_quant import W8A8Linear, quantize_qwen
from smoothquant.calibration import get_act_scales

from smoothquant.calibration import get_static_decoder_layer_scales

# model = AutoModelForCausalLM.from_pretrained('/root/autodl-tmp/Qwen-7B', 
#                                              device_map="auto",trust_remote_code=True).eval()

launcher_config = ProcessConfig(device_isolation=False)
backend_config = PyTorchConfig(model="gpt2", no_weights=True, device="cuda")
backend = PyTorchBackend(backend_config)
backend.pretrained_model=model
benchmark_config = InferenceConfig(memory=True,new_tokens=1000)
benchmark = InferenceBenchmark(benchmark_config)
benchmark.run(backend)
    
benchmark.report
#将得到的结果导出为 json 文件
report = benchmark.report
with open('benchmark_report_w8a8test.json', 'w') as json_file:
    json.dump(report.to_dict(), json_file, indent=4)
#将配置文件导出为 json 文件
backend_config = vars(backend.config)
benckmark_config = vars(benchmark.config)
if 'generate_kwargs' in benckmark_config and 'logits_processor' in benckmark_config['generate_kwargs']:
    benckmark_config['generate_kwargs'].pop('logits_processor')
    config={"backend": backend_config, "benchmark": benckmark_config,}
    with open('experiment_config.json', 'w') as json_file:
        json.dump(config, json_file, indent=4)


# In[ ]:





# In[ ]:




