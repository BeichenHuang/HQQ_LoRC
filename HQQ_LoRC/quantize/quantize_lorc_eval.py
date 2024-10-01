from safetensors.torch import save_file
from safetensors import safe_open
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from hqq.core.quantize import *
from hqq.models.hf.llama import LlamaHQQ as AutoHQQHFModel
import gc,os
import argparse
torch.cuda.empty_cache()
import numpy as np
import scipy.stats as stats

def UV_int8_dequantize(LoRC_weight_path, UV, layer_name):
    with safe_open(f"{LoRC_weight_path}/{UV}_int8_scale.safetensors", framework="pt", device="cuda") as f:
        scale = f.get_tensor(layer_name)
    with safe_open(f"{LoRC_weight_path}/{UV}_int8_zero.safetensors", framework="pt", device="cuda") as f:
        zero = f.get_tensor(layer_name)
    with safe_open(f"{LoRC_weight_path}/{UV}_int8_weight.safetensors", framework="pt", device="cuda") as f:
        weight = f.get_tensor(layer_name)
    dequantized_weight = (weight - zero) / scale #dequantize
    return dequantized_weight.half()

def Error_gen(model_path = '/u/yyuan6/hqq_lorc/llama/llama-3bit-quantized',
              quant = 'int8',
              lorc_path = None):
    average_L2 = {}
    average_rank = {}
    average_kurtosis_value = {}
    fpath = "/work/hdd/bcjw/yyuan6/hqq_lorc/llama/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
    model = AutoHQQHFModel.from_quantized(model_path)
    quant_list = ['v_proj']
    print(quant_list)
    for f_idx in range(1,3):
        fname = f"{fpath}/model-{f_idx:05}-of-00002.safetensors"
        with safe_open(fname, framework="pt", device="cuda") as f:
            for key in f.keys():
                # print(key)
                if not any(q in key for q in quant_list): continue
                # print(key)
                layer_name = key[:-len(".weight")]
                W_orig = f.get_tensor(key)
                layer_q = dict(model.named_modules())[layer_name]
                W_q = layer_q.dequantize()

                if lorc_path is not None:
                    print(f"adding low rank to {layer_name} ..")
                    U = UV_int8_dequantize(lorc_path, 'U', layer_name).to('cuda')
                    V = UV_int8_dequantize(lorc_path, 'V', layer_name).to('cuda')
                    W_q = W_q.to('cuda') + (U @ V)

                U, S, V = torch.linalg.svd(W_orig.float() - W_q.float(), full_matrices=False)  #do SVD to error matrix
                
                error_matrix = W_orig.float() - W_q.float()
                l2norm = torch.norm(error_matrix, p='fro')
                num_elements = error_matrix.numel()
                l2norm = l2norm / torch.sqrt(torch.tensor(num_elements, dtype=torch.float))

                p_half_norm = torch.norm(error_matrix, p=0.5)

                mean = torch.mean(error_matrix)
                variance = torch.var(error_matrix, unbiased=False)
                fourth_moment = torch.mean((error_matrix - mean) ** 4)
                kurt = fourth_moment / (variance ** 2)
                
                kurtosis = kurt - 3.0
                kurtosis_value = kurtosis.item()

                num_elements = error_matrix.numel()
                normalized_p_half_norm = p_half_norm / (num_elements ** (1 / 0.5))

                # Define a tolerance to treat very small singular values as zero
                max_S = torch.max(S)
                tolerance = max_S * 0.5
                error_matrix_rank = torch.sum(S > tolerance)
                print(f"{key}: Rank: {error_matrix_rank.item()}, L2-norm:{l2norm.item()}, L0.5-norm:{normalized_p_half_norm.item()}, kurtosis_value:{kurtosis_value}")

                if key[-17:-7] in average_L2:
                    average_L2[key[-17:-7]] += l2norm.item()
                    average_rank[key[-17:-7]] += error_matrix_rank.item()
                    average_kurtosis_value[key[-17:-7]] += kurtosis_value 
                else:
                    average_L2[key[-17:-7]] = l2norm.item()
                    average_rank[key[-17:-7]] = error_matrix_rank.item()
                    average_kurtosis_value[key[-17:-7]] = kurtosis_value

                del W_orig,W_q
                S = torch.diag(S)

                del U,S,V
                gc.collect()
    for key, value in average_L2.items():
        print(f"{key}: L2norm={value / 32}, rank={average_rank[key]/32}, kurtosis_value={average_kurtosis_value[key]/32}")


def main():
    parser = argparse.ArgumentParser(description="error gen")
    
    parser.add_argument('--model_path', type=str,nargs='?', default='/work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized', help="model_path")
    parser.add_argument('--lorc_path', nargs='?', default=None, help="model_path")
    parser.add_argument('--error_quant', type = str, nargs='?',default='int8')
    args = parser.parse_args()
    print(f"orig model path: {args.model_path}")
    print(f"quant to: {args.error_quant}")
    print(f"lorc_path: {args.lorc_path}")
    Error_gen(
              args.model_path,
              args.error_quant,
              args.lorc_path)

if __name__ == "__main__":
    main()



    