import torch
from hqq.core.quantize import *
from hqq.models.hf.mixtral import MixtralHQQ as AutoHQQHFModel
from hqq.engine.hf import AutoTokenizer
from transformers import AutoTokenizer
import gc
import sys
sys.path.append('/u/bhuang4/MoE_quant')
sys.path.append('/u/bhuang4/MoE_quant/evaluation')
from eval_perplexity import eval_perplexity
import time
torch.cuda.empty_cache()
gc.collect()

model_id       = "mistralai/Mixtral-8x7B-v0.1" 
device         = 'cuda:0'
quant_path     = '/u/yyuan6/hqq_lorc/mixtral/mixtral-4bit-quantized'

model = AutoHQQHFModel.from_quantized(quant_path, device="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
