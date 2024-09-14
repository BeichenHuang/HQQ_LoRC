import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.deepseek import MixtralHQQ as AutoHQQHFModel
from hqq.core.quantize import *


model_id      = "deepseek-ai/deepseek-moe-16b-base" 
mytoken       = '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'
# cache_path    = '/scratch/bcjw/bhuang4/cache'
model_path    = '/u/yyuan6/hqq_lorc/deepseek-moe/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580'
compute_dtype = torch.float16
device        = "cuda"
# print(f"==Evaluating for the model , backend = marlin, {model_path}==")

model     = AutoModelForCausalLM.from_pretrained(model_path, token = mytoken, torch_dtype=compute_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id, token = mytoken) 

quant_config = BaseQuantizeConfig(nbits=4, group_size=64, quant_scale=False, quant_zero=False,axis=1) 

AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)


dir_s = f"/u/yyuan6/hqq_lorc/deepseek-moe/deepseek-moe-16b-4bit-quantized"
import os
if not os.path.exists(dir_s):
    os.makedirs(dir_s)

AutoHQQHFModel.save_quantized(model,dir_s)

