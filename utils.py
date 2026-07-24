import pathlib
import functools
import time
import os
from tqdm import tqdm
import numpy as np
import random
import torch
import json
from module.config import UNK_IDX, PAD_IDX

cur_dir = pathlib.Path(__file__).parent


def timing(func):
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        return_data = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print(
            f'function [{func.__name__}] finished in {int(elapsedTime * 1000)} ms'
        )
        return return_data

    return newfunc

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:2'
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

@timing
def load_embedding(word_embedding_path, t='dict'):
    embeddings = None
    if t == 'dict':
        embeddings = {}
        f = open(
            word_embedding_path, 'r', encoding='utf-8'
        )
        for line in f.readlines():
            values = line.split()
            word = str(values[0])
            embeddings[word] = np.asarray(values[1:], dtype='float32')
        f.close()
    elif t == 'np':
        embeddings = np.load(word_embedding_path).astype(np.float32)
    return embeddings

def padding_patent_text(title, abstract, claims, pad_val=0):
    max_len = 300
    max_claims = 30
    def padding(_in):
        if len(_in) < max_len:
            _in += [pad_val] * (max_len - len(_in))
        else:
            _in = _in[:max_len]
        return _in
    title = padding(title)
    abstract = padding(abstract)
    claims_len = len(claims)
    for i in range(claims_len):
        if i == max_claims:
            break
        claims[i] = padding(claims[i])
    if claims_len < max_claims:
        claims += [[pad_val] * max_len for _ in range(max_claims - len(claims))]
    else:
        claims = claims[:max_claims]
    return title, abstract, claims


def get_tokens_mapping(path, t='normal'):
    if t in ['company', 'patent']:
        idx2token = []
        token2idx = {}
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f.readlines()):
                token, _ = line.strip().split(',')
                token = int(token) if t == 'patent' else token
                idx2token.append(token)
                token2idx[token] = i
        return idx2token, token2idx
    if t == 'normal':
        idx2token = []
        token2idx = {}
        pre = 0
    elif t == 'glove':
        idx2token = ['<pad>', '<unk>']
        token2idx = {'<pad>': PAD_IDX, '<unk>': UNK_IDX}
        pre = 2
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f.readlines()):
            token = line.strip()
            idx2token.append(token)
            token2idx[token] = i + pre
    return idx2token, token2idx


def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.5)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256,1024,block_mem)
    del x

def load_bert_data(data_path, tokenizer):
    input_ids = []  # input char ids
    input_types = []  # segment ids
    input_masks = []  # attention mask
    label = []  # 标签
    max_len = 75  # 也称为 max_len
    with open(data_path, "r", encoding='utf-8') as f:
        for i, l in tqdm(enumerate(f.readlines())):
            # try:
            #     x1, y = l.strip().split('\t')
            # except ValueError:
            #     print(l.strip().split('\t'))
            data = json.loads(l)
            x1 = ''.join(data['c'])
            y = data['label']
            x1 = tokenizer.tokenize(x1)
            tokens = ["[CLS]"] + x1

            # 得到input_id, seg_id, att_mask
            ids = tokenizer.convert_tokens_to_ids(tokens)
            types = [0] * (len(ids))
            masks = [1] * len(ids)
            # 短则补齐，长则切断
            if len(ids) < max_len:
                types = types + [1] * (max_len - len(ids))  # mask部分 segment置为1
                masks = masks + [0] * (max_len - len(ids))
                ids = ids + [0] * (max_len - len(ids))
            else:
                types = types[:max_len]
                masks = masks[:max_len]
                ids = ids[:max_len]
            input_ids.append(ids)
            input_types.append(types)
            input_masks.append(masks)
            #         print(len(ids), len(masks), len(types))
            assert len(ids) == len(masks) == len(types) == max_len
            label.append([int(y)])
    return input_ids, input_types, input_masks, label

if __name__ == '__main__':
    # get_aw_list()
    load_embedding('', 'mongodb')

