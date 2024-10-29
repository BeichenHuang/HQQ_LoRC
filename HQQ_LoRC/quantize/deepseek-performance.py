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
parser.add_argument('--dense_rank', type=int, required=True, help='Path to save the quantized model')
parser.add_argument('--exp_rank', type=int, required=True, help='Path to save the quantized model')
parser.add_argument('--fix_exp_rank', type=bool, required=False, default=False, help='Path to save the quantized model')
parser.add_argument('--iter_exp_only', type=bool, required=False, default=False, help='Path to save the quantized model')
parser.add_argument('--se_rank', type=int, required=False, default=0, help='Path to save the quantized model')
parser.add_argument('--model_path', type=str, nargs='?', default=False)
parser.add_argument('--error_path', type=str, help="Error_path")


args = parser.parse_args()

# Assign arguments to variables
iteration = args.iteration
dense_rank = args.dense_rank
exp_rank = args.exp_rank
se_rank = args.se_rank if args.se_rank != 0 else args.dense_rank

model_id       = "deepseek-ai/deepseek-moe-16b-base" 
mytoken       = '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'
compute_dtype = torch.float16
device        = "cuda"

fp16_model_path = "/work/hdd/bcjw/yyuan6/hqq_lorc/deepseek-moe/models--deepseek-ai--deepseek-moe-16b-base/snapshots/521d2bc4fb69a3f3ae565310fcc3b65f97af2580"
quant_model_path = args.model_path
lorc_path = args.error_path

tokenizer = AutoTokenizer.from_pretrained(model_id) 

quant_config = BaseQuantizeConfig(nbits=3, group_size=64, quant_scale=False, quant_zero=False,axis=1) 


ranks = {'self_attn': dense_rank, 'shared': se_rank, 'layers.0.mlp': se_rank}
if(args.fix_exp_rank):
    ranks['.mlp.experts.'] = exp_rank
else:
    ranks_tensor = torch.zeros(64)
    expert_frequencies_tensor_ds = torch.load('/work/hdd/bcjw/yyuan6/hqq_lorc/deepseek-moe/expert_frequencies_tensor.pt')
    for layer_index in tqdm(range(27)):
        freq = expert_frequencies_tensor_ds[layer_index]
        freq_sum = torch.sum(freq)
        for expert_index in range(len(freq)):
            rank = int(torch.round(freq[expert_index] / freq_sum * (exp_rank*len(freq))))   # Assign rank based on the weight
            ranks_tensor[expert_index] = rank
            ranks[f'layers.{layer_index + 1}.mlp.experts.{expert_index}.'] = rank
        print(ranks_tensor)

print(ranks)



save_path = f"{quant_model_path}-iter{iteration}"

print(f"generating model={save_path}, using lorc_path={lorc_path}-iter{iteration}")
model = AutoModelForCausalLM.from_pretrained(fp16_model_path,torch_dtype=compute_dtype, trust_remote_code=True)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, 
                                compute_dtype=compute_dtype, device=device, 
                                lorc_path=lorc_path, ranks=ranks, iters=iteration, iter_expert_only=args.iter_exp_only,
                                  lorc_dtype='int3_symm')
os.makedirs(save_path)
AutoHQQHFModel.save_quantized(model, save_path)

