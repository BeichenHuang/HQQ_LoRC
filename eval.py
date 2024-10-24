# sys.path.append("/home/LeiFeng/pingzhi/moe_quantize/optimum/")  # Add the path to Python's search path
# print(sys.path)

import argparse
import json
import os
import sys
sys.path.append('/u/bhuang4/mixtral_offloading/evaluation_more')
sys.path.append('/u/bhuang4/mixtral_offloading/evaluation_more/lm_eval')
from transformers import AutoTokenizer
from hqq.core.quantize import *
from hqq.models.hf.mixtral import MixtralHQQ as AutoHQQHFModel
from hqq.engine.hf import AutoTokenizer
# from auto_gptq import AutoGPTQForCausalLM_mixed_precision
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import initialize_tasks

LM_EVAL_TASK_KWARGS_DICT = {
    "winogrande": {"task": "winogrande", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "copa": {"task": "copa", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "openbookqa": {"task": "openbookqa", "num_fewshot": 0, "batch_size": 128, "metric": "acc_norm"},
    "hellaswag": {"task": "hellaswag", "num_fewshot": 0, "batch_size": 128, "metric": "acc_norm"},
    "lambada_openai": {"task": "lambada_openai", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "rte": {"task": "rte", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    "piqa": {"task": "piqa", "num_fewshot": 0, "batch_size": 128, "metric": "acc"},
    # "mmlu": {"task": "mmlu", "num_fewshot": 5, "batch_size": 16, "metric": "acc"},
    # "triQA": {"task": "triviaqa", "num_fewshot": 5, "batch_size": 16, "metric": "exact_match"},
}

cache_path     = '/scratch/bcjw/bhuang4/cache'
model_id       = "mistralai/Mixtral-8x7B-v0.1" 
device         = 'cuda:0'
model_path     = '/scratch/bcjw/bhuang4/mixtral/noIns_myQuant_HQQ_3bit_gs64' 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Perplexity for a model.")
    parser.add_argument("--model_path",type=str,default='/scratch/bcjw/bhuang4/mixtral/noIns_myQuant_HQQ_3bit_gs64')
    parser.add_argument("--LoRC_path",type=str,default ='/u/bhuang4/mixtral_offloading/HQQ_LoRC/mixtral_3bit/Error_r32')
    parser.add_argument("--do_LoRC",type=str,default ='True')
    args = parser.parse_args()
    print(f"do test for: {args.LoRC_path}")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = AutoHQQHFModel.from_quantized(args.model_path,LoRC_weight_path=args.LoRC_path,do_LoRC=args.do_LoRC,LoRC_dtype = 'int8',trust_remote_code=True)
    tokenizer    = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_path)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # model = AutoGPTQForCausalLM_mixed_precision.from_quantized(
        #     args.quant_model_path,
        #     low_cpu_mem_usage=True,
        #     device_map="auto",
        #     model_basename=quantized_model_file_base_name,
        #     use_safetensors=True,
        #     trust_remote_code=True,
        #     inject_fused_mlp=False,
        #     inject_fused_attention=False,
        #     # disable_exllama=args.disable_exllama,
        # )
    # save_file_path = '/u/bhuang4/mixtral_offloading/evaluation_more/test_result.json'
    save_file_path = os.path.join("/u/bhuang4/mixtral_offloading/evaluation_more",
                                  f"eval_result_{args.LoRC_path.split('/')[-1]}.json")
    all_metrics = {}
    if os.path.exists(save_file_path):
        with open(save_file_path, 'r') as file:
            all_metrics = json.load(file)

    # if args.proxy:
    #     LM_EVAL_TASK_KWARGS_DICT.pop("mmlu")
    #     print("Skip MMLU for proxy benchmark as it is too large.")

    for task_kwargs in LM_EVAL_TASK_KWARGS_DICT.values():
        print(f"Evaluating task: {task_kwargs['task']}")
        task_name = task_kwargs["task"]
        lm = HFLM(
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=task_kwargs["batch_size"],
        )
        initialize_tasks(verbosity="ERROR")
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=task_name,
            num_fewshot=task_kwargs["num_fewshot"],
            batch_size=task_kwargs["batch_size"],
            log_samples=False,
        )
        metric = task_kwargs["metric"]
        for key, value in results["results"][task_name].items():
            if key.startswith(metric + ","):
                all_metrics[f"{task_name}_{metric}"] = value

        with open(save_file_path, 'w') as file:
            json.dump(all_metrics, file, indent=4)

    print(">>>>> Results <<<<<")
    # if args.is_quantized:
    #     print(f"Quantization on {args.model_name} from {args.quant_model_path}")
    # else:
    #     print(f"No quantization on {args.model_name}")
    average = sum(v for v in all_metrics.values()) / len(all_metrics)
    all_metrics["average"] = average
    print(f"Metrics: {all_metrics}")