from safetensors.torch import save_file
from safetensors import safe_open
import torch
from hqq.core.quantize import *
from hqq.models.hf.llama import LlamaHQQ as AutoHQQHFModel
import gc,os
import argparse
torch.cuda.empty_cache()
import numpy as np


# affine quantization
def full_to_int8(tensor_in):
    max_val, _ = torch.max(tensor_in, dim=1, keepdim=True)
    min_val, _ = torch.min(tensor_in, dim=1, keepdim=True)
    max_min = max_val - min_val
    max_min[max_min==0] = 255  #deal with the case max = min
    scale = 255 / max_min
    zero = - torch.round(scale * min_val) - 128  
    tensor_int8 = torch.round(tensor_in * scale + zero).to(torch.int8)
    return  scale,zero,tensor_int8


def pack_3bit_32(W_q_in: Tensor) -> Tensor:
    W_q = torch.zeros(
        [int(10 * np.ceil(W_q_in.shape[0] / 10.0)), W_q_in.shape[1]],
        device="cuda",
        dtype=int32,
    )
    W_q[: len(W_q_in)] = W_q_in
    _step = int(len(W_q) / 10)

    W_q = (
        (W_q[:_step] << 27)
        | (W_q[1 * _step : 2 * _step] << 24)
        | (W_q[2 * _step : 3 * _step] << 21)
        | (W_q[3 * _step : 4 * _step] << 18)
        | (W_q[4 * _step : 5 * _step] << 15)
        | (W_q[5 * _step : 6 * _step] << 12)
        | (W_q[6 * _step : 7 * _step] << 9)
        | (W_q[7 * _step : 8 * _step] << 6)
        | (W_q[8 * _step : 9 * _step] << 3)
        | (W_q[9 * _step : 10 * _step])
    )
    return W_q

def full_to_int3(tensor_in):
    max_val, _ = torch.max(tensor_in, dim=1, keepdim=True)
    min_val, _ = torch.min(tensor_in, dim=1, keepdim=True)
    max_min = max_val - min_val
    max_min[max_min==0] = 7  #deal with the case max = min
    scale = 7 / max_min
    zero = - torch.round(scale * min_val)
    tensor_int8 = torch.round(tensor_in * scale + zero).to(torch.int32)
    tensor_packed = pack_3bit_32(tensor_int8)
    return  scale,zero,tensor_packed

def full_to_int3_symmetric(tensor_in):
    scale, _ = torch.max(tensor_in, dim=1, keepdim=True)
    tensor_int8 = torch.round(tensor_in * 7/(2*scale) ) + 4
    tensor_int8 = torch.clamp(tensor_int8,0,7).to(torch.int32)
    tensor_packed = pack_3bit_32(tensor_int8)
    return  scale,tensor_packed

def Error_gen(save_path, 
              exp_rank = 8,
              attn_rank = 8,
              model_path = '/u/yyuan6/hqq_lorc/llama/llama-3bit-quantized',
              quant = 'int8',
              low_rank_only = False):
    print(f"==== Doing Error_gen for exp_rank {exp_rank}, attn_rank {attn_rank}====")
    all_U_h_half = {}
    all_V_h_half = {}

    all_U_h_q_weight = {}
    all_U_h_q_scale = {}
    all_U_h_q_zero = {}
    all_V_h_q_weight = {}
    all_V_h_q_scale = {}
    all_V_h_q_zero = {}
    average_L2 = {}
    average_rank = {}


    # count = torch.load('/u/bhuang4/mixtral_offloading/HQQ_LoRC/routing-count.pt')
    # _, indices = count.sort(dim=1, descending=True)
    # rank_list = [8,8,8,8,8,8,16,16]

    fpath = "/u/yyuan6/hqq_lorc/llama/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9"
    model = AutoHQQHFModel.from_quantized(model_path)
    # quant_list = ["mlp","self_attn"]
    quant_list = ['proj']
    print(quant_list)
    for f_idx in range(1,3):
        fname = f"{fpath}/model-{f_idx:05}-of-00002.safetensors"
        with safe_open(fname, framework="pt", device="cuda") as f:
            for key in f.keys():
                # print(key)
                if not any(q in key for q in quant_list): continue
                # print(key)
                layer_name = key[:-len(".weight")]
                W_orig = f.get_tensor(key)    #get the unquanzited weight
                layer_q = dict(model.named_modules())[layer_name]
                W_q = layer_q.dequantize()    #get the HQQ quntized weight
                U, S, V = torch.linalg.svd(W_orig.float() - W_q.float(), full_matrices=False)  #do SVD to error matrix
                
                error_matrix = W_orig.float() - W_q.float()
                l2norm = torch.norm(error_matrix, p='fro')
                num_elements = error_matrix.numel()
                l2norm = l2norm / torch.sqrt(torch.tensor(num_elements, dtype=torch.float))

                p_half_norm = torch.norm(error_matrix, p=0.5)


                num_elements = error_matrix.numel()
                normalized_p_half_norm = p_half_norm / (num_elements ** (1 / 0.5))

                # Define a tolerance to treat very small singular values as zero
                max_S = torch.max(S)
                tolerance = max_S * 0.5
                error_matrix_rank = torch.sum(S > tolerance)
                print(f"{key}: Rank: {error_matrix_rank.item()}, L2-norm:{l2norm.item()}, L0.5-norm:{normalized_p_half_norm.item()}")

                mean_error = torch.mean(torch.abs(error_matrix))
                std_error = torch.std(torch.abs(error_matrix))
                cv = std_error / mean_error
                print("Coefficient of Variation (CV):", cv.item())

                if key[-17:-7] in average_L2:
                    average_L2[key[-17:-7]] += l2norm.item()
                    average_rank[key[-17:-7]] += error_matrix_rank.item()
                else:
                    average_L2[key[-17:-7]] = l2norm.item()
                    average_rank[key[-17:-7]] = error_matrix_rank.item()

                del W_orig,W_q
                S = torch.diag(S)

                if 'mlp' in key:
                    rank = exp_rank
                else:
                    rank = attn_rank

                # if 'experts' in key:
                #     layer_pattern = r"layers.(\d+)"
                #     exp_pattern = r"experts.(\d+)"
                #     layer_idx = int(re.search(layer_pattern, key).group(1))
                #     exp_idx = int(re.search(exp_pattern, key).group(1))
                #     rank = rank_list[indices[layer_idx,exp_idx]]
                # else:
                #     rank = 8
                    
                # if 'layers.0.' in key:
                #     rank = 64
                # elif 'layers.1.' in key:
                #     rank = 64
                # elif 'layers.2.' in key:
                #     rank = 64
                # elif 'layers.3.' in key:
                #     rank = 64
                # elif 'experts' in key:
                #     rank = 8
                # elif 'self_attn' in key:
                #     rank = 64
                # else:
                #     print("something wrong")

                # max_val = torch.max(S)
                # if torch.sum(S > max_val*0.8).item() < 64:
                #     rank = 64
                # else:
                #     print("jump this layer")
                #     U_shape = U.shape[0]
                #     V_shape = V.shape[1]

                #     all_U_h_int8_weight[layer_name] = torch.zeros(U_shape, 1).to('cpu')
                #     all_U_h_int8_scale[layer_name] = torch.ones(U_shape, 1).to('cpu')
                #     all_U_h_int8_zero[layer_name] = torch.zeros(U_shape, 1).to('cpu')

                #     all_V_h_int8_weight[layer_name] = torch.zeros(1,V_shape).to('cpu')
                #     all_V_h_int8_scale[layer_name] = torch.ones(1,V_shape).to('cpu')
                #     all_V_h_int8_zero[layer_name] = torch.zeros(1,V_shape).to('cpu')
                #     continue
                print(f"rank is:{rank}")
                U_h = U[:,:rank] @ torch.sqrt(S[:rank,:rank]) # calculate the U_h and V_h according to rank
                V_h = torch.sqrt(S[:rank,:rank]) @ V[:rank,:] 
                all_U_h_half[layer_name] = U_h.half().to('cpu')
                all_V_h_half[layer_name] = V_h.half().to('cpu')
                if quant == 'int8':
                    U_h_scale,U_h_zero,U_h_q = full_to_int8(U_h)
                    V_h_scale,V_h_zero,V_h_q = full_to_int8(V_h)
                elif quant =='int3':
                    U_h_scale,U_h_zero,U_h_q = full_to_int3(U_h)
                    V_h_scale,V_h_zero,V_h_q = full_to_int3(V_h)
                elif quant == 'int3_symm':
                    U_h_scale,U_h_q = full_to_int3_symmetric(U_h)
                    V_h_scale,V_h_q = full_to_int3_symmetric(V_h)
                else:
                    print('error quant type wrong')
                all_U_h_q_weight[layer_name] = U_h_q.to('cpu')
                all_U_h_q_scale[layer_name] = U_h_scale.to('cpu')
                
                all_V_h_q_weight[layer_name] = V_h_q.to('cpu')
                all_V_h_q_scale[layer_name] = V_h_scale.to('cpu')
                
                if quant != 'int3_symm':
                    all_U_h_q_zero[layer_name] = U_h_zero.to('cpu')
                    all_V_h_q_zero[layer_name] = V_h_zero.to('cpu')
                # E_h = U_h @ V_h
                # E_h_half  = E_h.half()

                # scale,zero,E_h_int8 = full_to_int8(E_h,group_size)
                # all_E_half[layer_name] = E_h_half.to('cpu')

                # all_E_int8_weight[layer_name] = E_h_int8.to('cpu')
                # all_E_int8_scale[layer_name] = scale.to('cpu')
                # all_E_int8_zero[layer_name] = zero.to('cpu')
                del U,S,V,U_h,V_h
                gc.collect()
                # print(f"Layer: {layer_name} done")
            for key, value in average_L2.items():
                print(f"{key}: L2norm={value / 32}, rank={average_rank[key]/32}")

    os.makedirs(save_path,exist_ok = True)
    if quant == 'int8':
        save_file(all_U_h_q_weight, f"{save_path}/U_int8_weight.safetensors")
        save_file(all_U_h_q_scale, f"{save_path}/U_int8_scale.safetensors")
        save_file(all_U_h_q_zero, f"{save_path}/U_int8_zero.safetensors")
        save_file(all_V_h_q_weight, f"{save_path}/V_int8_weight.safetensors")
        save_file(all_V_h_q_scale, f"{save_path}/V_int8_scale.safetensors")
        save_file(all_V_h_q_zero, f"{save_path}/V_int8_zero.safetensors")
        print(">>int8 saved<<")
    elif quant == "int3":
        save_file(all_U_h_q_weight, f"{save_path}/U_int3_weight.safetensors")
        save_file(all_U_h_q_scale, f"{save_path}/U_int3_scale.safetensors")
        save_file(all_U_h_q_zero, f"{save_path}/U_int3_zero.safetensors")
        save_file(all_V_h_q_weight, f"{save_path}/V_int3_weight.safetensors")
        save_file(all_V_h_q_scale, f"{save_path}/V_int3_scale.safetensors")
        save_file(all_V_h_q_zero, f"{save_path}/V_int3_zero.safetensors")
        print(">>int3 saved<<")
    elif quant == "int3_symm":
        save_file(all_U_h_q_weight, f"{save_path}/U_int3_symm_weight.safetensors")
        save_file(all_U_h_q_scale, f"{save_path}/U_int3_symm_scale.safetensors")
        save_file(all_V_h_q_weight, f"{save_path}/V_int3_symm_weight.safetensors")
        save_file(all_V_h_q_scale, f"{save_path}/V_int3_symm_scale.safetensors")
        print(">>int3_symm saved<<")
    # save_file(all_U_h_half, f"{save_path}/U_r{rank}_half.safetensors")
    # save_file(all_V_h_half, f"{save_path}/V_r{rank}_half.safetensors")



def main():
    parser = argparse.ArgumentParser(description="error gen")
    
    # 添加参数
    parser.add_argument('--exp_rank', type=int, nargs='?', default=8, help="exp_rank")
    parser.add_argument('--attn_rank', type=int, nargs='?', default=8, help="attn_rank")
    parser.add_argument('--model_path', type=str,nargs='?', default='/u/yyuan6/hqq_lorc/llama/llama-3bit-quantized', help="model_path")
    parser.add_argument('--error_quant',type = str, nargs='?',default='int8')
    parser.add_argument('--save_path',type = str)
    parser.add_argument('--low_rank_only',type = bool, nargs='?',default=False)
    # 解析参数
    args = parser.parse_args()
    print(f"exp rank: {args.exp_rank}")
    print(f"attn rank: {args.attn_rank}")
    print(f"orig model path: {args.model_path}")
    print(f"quant to: {args.error_quant}")
    print(f"low rank only: {args.low_rank_only}")
    # print(f"group_size: {args.group_size}")
    Error_gen(args.save_path,
              args.exp_rank,
              args.attn_rank,
              args.model_path,
              args.error_quant,
              args.low_rank_only)

if __name__ == "__main__":
    main()



    