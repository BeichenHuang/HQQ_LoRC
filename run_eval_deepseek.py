import sys
sys.path.append("/u/yyuan6/hqq_lorc/HQQ_LoRC/HQQ_LoRC")
import torch
from hqq.core.quantize import *
from HQQ_LoRC.hqq.models.hf.deepseek import MixtralHQQ as AutoHQQHFModel
from hqq.engine.hf import AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gc
import sys
from eval_perplexity import eval_perplexity
import time
torch.cuda.empty_cache()
gc.collect()


model_id       = "deepseek-ai/deepseek-moe-16b-base" 
device         = 'cuda:0'

# quant_path     = '/u/yyuan6/hqq_lorc/deepseek-moe/deepseek-moe-16b-3bit-quantized'
# model = AutoHQQHFModel.from_quantized(quant_path, device="cuda:0")

model_path = "/u/yyuan6/hqq_lorc/deepseek-moe/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

begin = time.time()
eval_perplexity(model,tokenizer,f"{model_path}/wikitext2_perplexity_result.pickle",save_flag=0)
end = time.time()
print(f"taking {end - begin}")