import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hqq.models.hf.mixtral import MixtralHQQ as AutoHQQHFModel
from hqq.core.quantize import *
from error_gen_llama import Error_gen
from tqdm import tqdm

# Define command-line arguments
parser = argparse.ArgumentParser(description="Quantize a LLaMA model.")
parser.add_argument('--iteration', type=int, required=True, help='Path to save the quantized model')


args = parser.parse_args()

# Assign arguments to variables
iteration = args.iteration

model_id       = "mistralai/Mixtral-8x7B-v0.1" 
mytoken       = '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'
compute_dtype = torch.float16
device        = "cuda"

fp16_model_path = "/work/hdd/bcjw/yyuan6/hqq_lorc/mixtral/models--mistralai--Mixtral-8x7B-v0.1/snapshots/ffe1a706bacbd5abddc5ff99432ee38f7e0662fb"
quant_model_path = '/work/hdd/bcjw/yyuan6/hqq_lorc/mixtral/mixtral-3bit'
lorc_path = '/work/hdd/bcjw/yyuan6/hqq_lorc/mixtral/a1024efreqadaptive'

tokenizer = AutoTokenizer.from_pretrained(model_id) 

quant_config = BaseQuantizeConfig(nbits=3, group_size=64, quant_scale=False, quant_zero=False,axis=1) 


ranks = {'self_attn': 1024}
exp_avg_rank = 256
num_layers = 32

ranks_tensor = torch.zeros(num_layers)
expert_frequencies_tensor_ds = torch.load('/work/hdd/bcjw/yyuan6/hqq_lorc/mixtral/expert_frequencies_tensor.pt')
for layer_index in tqdm(range(num_layers)):
    freq = expert_frequencies_tensor_ds[layer_index]
    freq_sum = torch.sum(freq)
    for expert_index in range(len(freq)):
        rank = int(torch.round(freq[expert_index] / freq_sum * (exp_avg_rank * len(freq))))   # Assign rank based on the weight
        ranks_tensor[expert_index] = rank
        ranks[f'layers.{layer_index}.block_sparse_moe.experts.{expert_index}.'] = rank
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

