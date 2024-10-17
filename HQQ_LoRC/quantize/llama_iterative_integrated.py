import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hqq.models.hf.llama import LlamaHQQ as AutoHQQHFModel
from hqq.core.quantize import *
from error_gen_llama import Error_gen

# Define command-line arguments
parser = argparse.ArgumentParser(description="Quantize a LLaMA model.")
parser.add_argument('--fp16_model_path', type=str, required=True, help='Path to the pretrained model')
parser.add_argument('--quant_model_path', type=str, required=True, help='Path to the pretrained model')
parser.add_argument('--lorc_path', type=str, required=True, help='Path to the low-rank compressed weights')
parser.add_argument('--iteration', type=int, required=True, help='Path to save the quantized model')
parser.add_argument('--exp_rank', type=int, required=False, default=256, help='Path to save the quantized model')
parser.add_argument('--attn_rank', type=int, required=False, default=256, help='Path to save the quantized model')
parser.add_argument('--error_quant',type = str, nargs='?',default='int8')
parser.add_argument('--low_rank_only',type = bool, nargs='?',default=False)
parser.add_argument('--print_info',type = bool, nargs='?',default=False)

args = parser.parse_args()

# Assign arguments to variables
iteration = args.iteration

model_id      = "meta-llama/Llama-2-7b-hf" 
mytoken       = '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'
compute_dtype = torch.float16
device        = "cuda"

tokenizer = AutoTokenizer.from_pretrained(model_id) 

quant_config = BaseQuantizeConfig(nbits=3, group_size=64, quant_scale=False, quant_zero=False,axis=1) 


ranks = {'q_proj': 256, 'k_proj': 256, 'v_proj': 256, 'o_proj': 256, 'mlp':256}

save_path = f"{args.quant_model_path}-iter{iteration}"

print(f"generating model={save_path}, using lorc_path={args.lorc_path}-iter{iteration}")
model = AutoModelForCausalLM.from_pretrained(args.fp16_model_path,torch_dtype=compute_dtype, trust_remote_code=True)
AutoHQQHFModel.quantize_model(model, quant_config=quant_config, 
                                compute_dtype=compute_dtype, device=device, 
                                lorc_path=args.lorc_path, ranks=ranks, iters=iteration)
os.makedirs(save_path)
AutoHQQHFModel.save_quantized(model, save_path)

