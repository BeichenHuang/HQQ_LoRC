import torch
import pandas as pd
import pickle
from tqdm import tqdm
import time
import gc

# import gpustat
# def print_gpu_memory():

#     # 获取所有GPU的状态
#     gpu_stats = gpustat.new_query()
    
#     # 输出每个GPU的详细信息
#     for gpu in gpu_stats.gpus:
#         print(f"  显存使用情况: {gpu.memory_used} MB / {gpu.memory_total} MB")


def eval_perplexity(model,tokenizer,save_name,save_flag):

    device = torch.device("cuda:0")
    fname = "/projects/bcvk/bhuang4/dataset/wikitext2/wikitext2_test-00000-of-00001.parquet"
    df = pd.read_parquet(fname)
    texts = df['text'].tolist()
    encodings = tokenizer("\n\n".join(texts), return_tensors="pt")
    # print("size")
    # print(encodings.input_ids.size())
    max_length = 2048
    stride = 512
    seq_len = encodings.input_ids.size(1)

    # max_length = 32
    # stride = 16
    # seq_len = 512
    
    test_begin = time.time()
    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
    # for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        # print(f"begin:{begin_loc}, end:{end_loc}")
        # print_gpu_memory()
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        # outputs = model(input_ids, labels=target_ids)
        # del outputs
        # torch.cuda.empty_cache()
        # gc.collect()
        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
            neg_log_likelihood = outputs.loss
       
        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    test_end = time.time()
    print(f"calculate perplexity time:{test_end - test_begin}s")
    ppl = torch.exp(torch.stack(nlls).mean())
    print('\n ppl is: ', ppl.item())
    if(save_flag):
        with open(save_name, 'wb') as file:
            pickle.dump(ppl, file)



