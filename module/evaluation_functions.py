from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from .time_track import time_desc_decorator


@time_desc_decorator('auc_scores_at_k')
def leave_one_auc_scores_at_k(samples, pred, neg_type, k_s=None):
    """
    :param samples: train/val/test samples, shape: [sample_length, 6]
    :param pred: train/val/test predictions, shape: [sample_length, 2]
    :param k_s: top_k_list
    :return: precision_s, recall_s, f1_s
    """
    if not samples:
        return 0, 0, 0, 0
    if k_s is None:
        k_s = [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    pairs_samples = {}
    pairs_preds = {}
    if neg_type == 0:
        n_i, p_i, p_j = 0, 1, 2
    elif neg_type == 1:
        n_i, p_i, p_j = 1, 0, 2
    elif neg_type == 2:
        n_i, p_i, p_j = 2, 0, 1
    else:
        raise NotImplementedError
    for index, sample in enumerate(samples):
        p_idx_1 = sample[p_i]
        p_idx_2 = sample[p_j]
        # Record positive sample subscript+label+predicted value and negative sample subscript+label+predicted value for each pair
        if (p_idx_1, p_idx_2) in pairs_samples:
            pairs_samples[p_idx_1, p_idx_2].add((sample[n_i], 1))
            pairs_samples[p_idx_1, p_idx_2].add((sample[n_i+3], 0))
            pairs_preds[p_idx_1, p_idx_2][sample[n_i], 1] = pred[index][0]
            pairs_preds[p_idx_1, p_idx_2][sample[n_i+3], 0] = pred[index][1]
        else:
            pairs_samples[p_idx_1, p_idx_2] = set([(sample[n_i], 1)])
            pairs_samples[p_idx_1, p_idx_2].add((sample[n_i+3], 0))
            pairs_preds[p_idx_1, p_idx_2] = {(sample[n_i], 1): pred[index][0]}
            pairs_preds[p_idx_1, p_idx_2][sample[n_i+3], 0] = pred[index][1]

    # First, for each firm pair, the sample is de-duplicated, then sorted, and then precision, recall, f1 at k is calculated
    precision_s = {k: 0 for k in k_s}
    recall_s = {k: 0 for k in k_s}
    f1_s = {k: 0 for k in k_s}
    ndcg_s = {k: 0 for k in k_s}
    mrr_s = {k: 0 for k in k_s}
    hr_s = {k: 0 for k in k_s}
    pair_count = 0
    auc = 0
    for key, result in pairs_samples.items():
        result = list(result)
        labels, preds = [], []
        for x in result:
            labels.append(x[1])
            preds.append(pairs_preds[key][x])
        auc += roc_auc_score(labels, preds)
        relevant = sum(labels)
        # result is a list with the elements [rec_item, item_label] sorted in descending order by the predicted score
        result.sort(key=lambda x: pairs_preds[key][x], reverse=True)
        for k in k_s:
            # How many positive examples are in top k
            pos_num = sum([x[1] for x in result[:k]])
            if pos_num == 0:
                continue
            prec = pos_num / min(k, len(result))
            precision_s[k] += prec
            rec = pos_num / relevant
            recall_s[k] += rec
            f1_s[k] += 2 * prec * rec / (prec + rec + 1e-7)
            # ndcg
            rec_labels = np.array([x[1] for x in result[:k]])
            rank = np.arange(1, min(len(result), k) + 1, 1)
            idcgs = 1. / np.log2(rank + 1)
            idcg = sum(idcgs[:pos_num])
            dcgs = idcgs[np.where(rec_labels == 1)]
            dcg = sum(dcgs)
            ndcg_s[k] += dcg / idcg
            # mrr
            ranks = np.where(rec_labels == 1)[0] + 1
            rranks = 1 / ranks
            mrr_s[k] += sum(rranks) / rranks.shape[0]
            # hit rate
            hr_s[k] += 1 if pos_num > 0 else 0
        pair_count += 1
    for k in k_s:
        precision_s[k] /= pair_count
        recall_s[k] /= pair_count
        f1_s[k] /= pair_count
        ndcg_s[k] /= pair_count
        mrr_s[k] /= pair_count
        hr_s[k] /= pair_count
    auc /= pair_count
    return auc, precision_s, recall_s, f1_s, ndcg_s, mrr_s, hr_s
