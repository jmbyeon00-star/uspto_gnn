# import torch
# import random
# import pandas as pd
# from torch.utils.data import Dataset
# from utils import get_tokens_mapping

from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import json
import torch
import random
import pandas as pd
try:
    from .utils import get_tokens_mapping
except ImportError:
    from utils import get_tokens_mapping
    
class MANTFDataset(Dataset):
    """
    docstring
    dataset:ids, types, masks, graph_node_ids
    """
    def __init__(self, patent_idx_path, company_idx_path, dataset_path, neg_types, train_mode=True):
        _, patent2idx = get_tokens_mapping(patent_idx_path, 'patent')
        _, company2idx = get_tokens_mapping(company_idx_path, 'company')
        # _, token2idx = get_tokens_mapping(token_map_path, 'glove')
        samples = pd.read_csv('{}.csv'.format(dataset_path))
        self.p_plaintiff, self.p_defendant = [], []
        self.n_plaintiff, self.n_defendant = [], []
        self.p_patent, self.n_patent = [], []
        self.neg_types = []
        for neg in neg_types:
            for _, sample in samples[samples['neg_type']==neg].iterrows():
            # for sample in tqdm(data_dict[dataset].find({'neg_type': neg}), total=data_dict[dataset].count_documents({'neg_type': neg})):
                try:
                    p_plaintiff = company2idx[sample['pos_pla_id']]
                except:
                    print(sample)
                p_defendant = company2idx[sample['pos_def_id']]
                p_patent = patent2idx[sample['pos_pat_id']]
                n_plaintiff, n_defendant, n_patent = p_plaintiff, p_defendant, p_patent
                sample_neg_type = sample['neg_type']
                if sample_neg_type == 0:
                    n_plaintiff = company2idx[sample['neg_id']]
                elif sample_neg_type == 1:
                    n_defendant = company2idx[sample['neg_id']]
                elif sample_neg_type == 2:
                    n_patent = patent2idx[int(sample['neg_id'])]
                # 0 1 neg type 
                # The positive and negative sample pairs are randomly dropped to half, ensuring that the ratio of three tasks is 1:1:2
                if train_mode and sample_neg_type in [0, 1] and random.random() > 0.5:
                    continue
                self.p_plaintiff.append(p_plaintiff)
                self.p_defendant.append(p_defendant)
                self.p_patent.append(p_patent)
                self.n_plaintiff.append(n_plaintiff)
                self.n_defendant.append(n_defendant)
                self.n_patent.append(n_patent)
                self.neg_types.append(sample_neg_type)

    def __getitem__(self, index):
        return self.p_plaintiff[index], self.p_defendant[index], self.p_patent[index], \
                self.n_plaintiff[index], self.n_defendant[index], self.n_patent[index], \
                    self.neg_types[index]

    def __len__(self):
        return len(self.p_patent)
    

def collate(samples):
    p_pla_idx, p_def_idx, p_pat_idx, n_pla_idx, n_def_idx, n_pat_idx, neg_types = zip(*samples)
    p_pla_idx = torch.tensor(p_pla_idx, dtype=torch.long)
    p_def_idx = torch.tensor(p_def_idx, dtype=torch.long)
    p_pat_idx = torch.tensor(p_pat_idx, dtype=torch.long)

    n_pla_idx = torch.tensor(n_pla_idx, dtype=torch.long)
    n_def_idx = torch.tensor(n_def_idx, dtype=torch.long)
    n_pat_idx = torch.tensor(n_pat_idx, dtype=torch.long)

    neg_types = torch.tensor(neg_types, dtype=torch.long)
    return p_pla_idx, p_def_idx, p_pat_idx, n_pla_idx, n_def_idx, n_pat_idx, neg_types
