# usage

1. quantize the fp16 model using previous quantization script. For example, save to /path/to/llama-3bit-quantized.

2. The first low rank composition needs to be done manually, using `Error_gen.py` as before. Assume saving this to /path/to/uv-all256.

3. Do iterative optimization.
```
python3 llama_3bit_iterative.py --fp16_model_path /path/to/llama-3bit-fp16 --quant_model_path /path/to/llama-3bit-quantized --lorc_path /path/to/uv-all256 --iteration 5 --exp_rank 256 --attn_rank 256
```
Use the final iteration as the flag. The existing iters will be skipped.

An example:
```
(env310) ➜  quantize git:(deepseek) ✗ python3 llama_3bit_iterative.py --quant_model_path /work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized --lorc_path /work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256 --iteration 7 --fp16_model_path /work/hdd/bcjw/yyuan6/hqq_lorc/llama/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  4.47it/s]
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized-iterate1 exists. Skip.
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256-iterate1 exists. Skip.
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized-iterate2 exists. Skip.
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256-iterate2 exists. Skip.
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized-iterate3 exists. Skip.
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256-iterate3 exists. Skip.
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized-iterate4 exists. Skip.
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256-iterate4 exists. Skip.
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized-iterate5 exists. Skip.
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256-iterate5 exists. Skip.
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized-iterate6 exists. Skip.
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256-iterate6 exists. Skip.
generating model=/work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized-iterate7, using lorc_path=/work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256-iterate6
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 1245.56it/s]
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:40<00:00,  1.26s/it]
generating lorc=/work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256-iterate7, using model_path=/work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized-iterate7
==== Doing Error_gen for exp_rank 256, attn_rank 256====
/work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized-iterate7/config.json
/work/hdd/bcjw/yyuan6/hqq_lorc/HQQ_LoRC_gh/HQQ_LoRC/hqq/models/base.py:256: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  return torch.load(cls.get_weight_file(save_dir), map_location=map_location)
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 1024.75it/s]
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 32/32 [00:00<00:00, 1580.33it/s]
['proj']
Processing file 00001: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 241/241 [03:31<00:00,  1.14it/s]
Processing file 00002: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 82/82 [01:09<00:00,  1.18it/s]
Files Processing: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [04:40<00:00, 140.28s/it]
>>int8 saved to /work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256-iterate7<<
```
# evaluation
Use the same evaluation script as before, with iterated `model_path` and `error_path`.
```
python3 run_eval_HQQ_LoRC_llama.py --model_path /work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized-iterate7 --error_path /work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256-iterate7 --exp_rank 256 --attn_rank 256
```
`quantize_lorc_eval` is used to evaluate the error matrix norms for each parameter.
```
python3 quantize_lorc_eval.py --model_path /work/hdd/bcjw/yyuan6/hqq_lorc/llama/llama-3bit-quantized-iterate5 --lorc_path /work/hdd/bcjw/yyuan6/hqq_lorc/llama/all256-iterate5
```
# apply to another model

1. change `model_id`.
1. change `Error_gen` function to the corresponding model. 
