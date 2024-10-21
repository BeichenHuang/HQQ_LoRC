import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hqq.models.hf.deepseek import MixtralHQQ as AutoHQQHFModel
from hqq.core.quantize import *
from error_gen_llama import Error_gen
from tqdm import tqdm

# Define command-line arguments
parser = argparse.ArgumentParser(description="Quantize a LLaMA model.")
parser.add_argument('--iteration', type=int, required=True, help='Path to save the quantized model')


args = parser.parse_args()

# Assign arguments to variables
iteration = args.iteration

model_id       = "deepseek-ai/deepseek-moe-16b-base" 
mytoken       = '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'
compute_dtype = torch.float16
device        = "cuda"

fp16_model_path = "/work/hdd/bcjw/yyuan6/hqq_lorc/deepseek-moe/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
quant_model_path = '/work/hdd/bcjw/yyuan6/hqq_lorc/deepseek-moe/deepseek-3bit'
lorc_path = '/work/hdd/bcjw/yyuan6/hqq_lorc/deepseek-moe/a1024efreq_adaptive'

tokenizer = AutoTokenizer.from_pretrained(model_id) 

quant_config = BaseQuantizeConfig(nbits=3, group_size=64, quant_scale=False, quant_zero=False,axis=1) 


ranks = {'self_attn': 1024, 'shared':1024, 'layers.0.mlp':1024}
ranks_tensor = torch.zeros(64)
expert_frequencies_tensor_ds = torch.load('/work/hdd/bcjw/yyuan6/hqq_lorc/deepseek-moe/expert_frequencies_tensor.pt')
for layer_index in tqdm(range(27)):
    freq = expert_frequencies_tensor_ds[layer_index]
    freq_sum = torch.sum(freq)
    for expert_index in range(len(freq)):
        rank = int(torch.round(freq[expert_index] / freq_sum * (32*len(freq))))   # Assign rank based on the weight
        ranks_tensor[expert_index] = rank
        ranks[f'layers.{layer_index + 1}.mlp.experts.{expert_index}.'] = rank
    print(ranks_tensor)

print(ranks)



save_path = f"{quant_model_path}-iter{iteration}"

print(f"generating model={save_path}, using lorc_path={lorc_path}-iter{iteration}")
model = AutoModelForCausalLM.from_pretrained(fp16_model_path,torch_dtype=compute_dtype, trust_remote_code=True)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, 
                                compute_dtype=compute_dtype, device=device, 
                                lorc_path=lorc_path, ranks=ranks, iters=iteration)
os.makedirs(save_path)
AutoHQQHFModel.save_quantized(model, save_path)

