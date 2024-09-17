from safetensors import safe_open
import torch
from hqq.core.quantize import *
from hqq.models.hf.deepseek import MixtralHQQ as AutoHQQHFModel
from hqq.engine.hf import AutoTokenizer
import gc
import sys
from eval_perplexity import eval_perplexity
import time
torch.cuda.empty_cache()
gc.collect()
import argparse


model_id       = "deepseek-ai/deepseek-moe-16b-base" 
device         = 'cuda:0'
model_path     = '/u/yyuan6/hqq_lorc/deepseek-moe/deepseek-moe-16b-3bit-quantized' 


def main():
    parser = argparse.ArgumentParser(description="HQQ_LoRC")
    parser.add_argument('--Error_path', type=str, help="Error_path")
    parser.add_argument('--LoRC_dtype', type=str, default ='int8', help="LoRC_dtype")
    parser.add_argument('--exp_rank', type=int, help="exp_rank")
    parser.add_argument('--attn_rank', type=int, help="attn_rank")
    parser.add_argument('--low_rank_only',type = bool, nargs='?',default=False)

    args = parser.parse_args()
    print(f"model path: {model_path}")
    print(f"Error_path: {args.Error_path}")
    print(f"LoRC_dtype: {args.LoRC_dtype}")
    print(f"exp_rank: {args.exp_rank}")
    print(f"attn_rank: {args.attn_rank}")
    print(f"low_rank_only: {args.low_rank_only}")
    model = AutoHQQHFModel.from_quantized(model_path,LoRC_weight_path=args.Error_path,LoRC_dtype = args.LoRC_dtype,exp_rank=args.exp_rank,attn_rank=args.attn_rank,low_rank_only=args.low_rank_only)
    tokenizer    = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token =tokenizer.eos_token
    # prompt1 = "Write an essay about large language models."
    # warm_inputs = tokenizer(
    #         [prompt1],
    #         padding=True,
    #         add_special_tokens=True,
    #         return_tensors="pt",
    #     ).to(device)

    # print("doing warm up")
    # _ = model.generate(**warm_inputs, max_new_tokens=512, do_sample=True)
    # print("finish warm up")

    begin = time.time()
    eval_perplexity(model,tokenizer,f"{model_path}/wikitext2_perplexity_result.pickle",save_flag=0)
    end = time.time()
    print(f"taking {end - begin}")
    return

if __name__ == "__main__":
    main()


# model = AutoHQQHFModel.from_quantized(model_path,U_path=U_path,V_path=V_path)



# no_list = ["embed_tokens","rotary_emb","gate","act_fn","layernorm","norm","lm_head"]
# for name, module in model.named_modules():
#     if len(list(module.children())) == 0:  # 只打印没有子模块的模块
#         if any(word in name for word in no_list): continue
#         if name in no_list: continue
#         print(name)
#         print(dir(module))
#         break

