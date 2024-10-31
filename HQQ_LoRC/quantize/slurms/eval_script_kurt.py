from safetensors import safe_open
import torch
import sys
sys.path.append("/work/hdd/bcjw/yyuan6/hqq_lorc/HQQ_LoRC_gh/HQQ_LoRC")
from hqq.core.quantize import *
from hqq.models.hf.deepseek import MixtralHQQ as AutoHQQHFModel
from hqq.engine.hf import AutoTokenizer
import gc
from eval_perplexity import eval_perplexity
import time
torch.cuda.empty_cache()
gc.collect()
import argparse


model_id       = "deepseek-ai/deepseek-moe-16b-base" 
device         = 'cuda:0'


def main():
    parser = argparse.ArgumentParser(description="HQQ_LoRC")
    parser.add_argument('--model_path', type=str, help="Error_path")
    parser.add_argument('--Error_path', type=str, help="Error_path")
    parser.add_argument('--LoRC_dtype', type=str, default ='int8', help="LoRC_dtype")
    parser.add_argument('--exp_rank', type=int, help="exp_rank")
    parser.add_argument('--dense_rank', type=int, help="attn_rank")
    parser.add_argument('--se_rank', type=int, required=False, default=0, help='Path to save the quantized model')
    parser.add_argument('--fix_exp_rank', type=bool, required=False, default=False, help='Path to save the quantized model')


    args = parser.parse_args()
    model_path = args.model_path
    print(f"model path: {model_path}")
    print(f"Error_path: {args.Error_path}")
    print(f"LoRC_dtype: {args.LoRC_dtype}")
    print(f"exp_rank: {args.exp_rank}")
    print(f"attn_rank: {args.dense_rank}")
    se_rank = args.se_rank if args.se_rank != 0 else args.dense_rank

    dense_rank = args.dense_rank
    exp_rank = args.exp_rank

    ranks = {'self_attn': 512, 'shared':512, 'layers.0.mlp':512}
    kurtosis_rank = {}
    with open('/work/hdd/bcjw/yyuan6/hqq_lorc/kurtosis_deepseek.txt', 'r') as file:
        lines = file.readlines()
    for line in lines:
        parts = line.split(':')
        
        text = parts[0].replace('.weight', '').strip()
        if "self_attn" in text or "shared" in text or "layers.0.mlp" in text: 
            # rank = 512
            continue
        else:
            number = parts[1].strip()
            kurtosis = round(float(number))
            if kurtosis > 1: 
                rank = 716 
            else:
                rank = 0
            # rank = 0
            ranks[text] = rank
        kurtosis_rank[text] = torch.tensor(rank)

    model = AutoHQQHFModel.from_quantized(model_path,LoRC_weight_path=args.Error_path,
                                          LoRC_dtype = args.LoRC_dtype,
                                          ranks=ranks)


    
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

