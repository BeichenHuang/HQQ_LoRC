import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hqq.models.hf.mixtral import MixtralHQQ as AutoHQQHFModel
from hqq.core.quantize import *


model_id      = "mistralai/Mixtral-8x7B-v0.1" 
mytoken       = '...'
# cache_path    = '/scratch/bcjw/bhuang4/cache'
# model_path    = '/scratch/bcjw/bhuang4/mixtral_offloading/noIns_myQuant_HQQ_att4_exp4_axis1'
compute_dtype = torch.float16
device        = "cuda"
# print(f"==Evaluating for the model , backend = marlin, {model_path}==")

model     = AutoModelForCausalLM.from_pretrained(model_id, token = mytoken, torch_dtype=compute_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_id, token = mytoken) 

quant_config = BaseQuantizeConfig(nbits=3, group_size=64, quant_scale=False, quant_zero=False,axis=1) 

AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=compute_dtype, device=device)


dir_s = f"/projects/bcvk/bhuang4/mixtral_quant/noIns_myQuant_HQQ_3bit_gs64"
import os
if not os.path.exists(dir_s):
    os.makedirs(dir_s)

AutoHQQHFModel.save_quantized(model,dir_s)


