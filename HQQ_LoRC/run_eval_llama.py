import torch
from hqq.core.quantize import *
from hqq.models.hf.llama import LlamaHQQ as AutoHQQHFModel
from hqq.engine.hf import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gc
import sys
from eval_perplexity import eval_perplexity
import time
torch.cuda.empty_cache()
gc.collect()

model_id       = "meta-llama/Llama-2-7b-hf" 
device         = 'cuda:0'
quant_path     = '/u/yyuan6/hqq_lorc/llama/llama-3bit-quantized'

model = AutoHQQHFModel.from_quantized(quant_path, device="cuda:0")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

#model_path = "/u/yyuan6/hqq_lorc/llama/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
#model = AutoModelForCausalLM.from_pretrained(model_path).to(device)


begin = time.time()
eval_perplexity(model,tokenizer,f"{quant_path}/wikitext2_perplexity_result.pickle",save_flag=0)
end = time.time()
print(f"taking {end - begin}")