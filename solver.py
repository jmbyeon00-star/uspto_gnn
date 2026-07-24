import os
import re
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
from math import isnan

import warnings
warnings.filterwarnings('ignore')

from module.models import MANTF
from module import time_desc_decorator, seed_torch, device
from module.utils import load_embedding
from module.evaluation_functions import leave_one_auc_scores_at_k

from numpy import set_printoptions
set_printoptions(threshold=np.inf, linewidth=np.nan)

import torch
from torch import nn
from torch import optim
from torch import distributed
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score

model_dict = {
    "MANTF": MANTF,
}

optimizer_dict = {'RMSprop': optim.RMSprop,
                  'Adam': optim.Adam,
                  'Adagrad': optim.Adagrad,
                  'sgd': optim.SGD}

class Solver(object):
    def __init__(
        self, config, train_data_loader, val_data_loader, test_data_loader
    ):
        self.config = config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader
        self.writer = None
        self.model = None
        self.scheduler = None
        self.early_stop_epoch = 0
        # 0: 0, 1: 1, 2: 2, 3: all
        self.neg_type = sum(config.neg_types)
        if config.mode == 'case_study':
            pass

    def optuna_build(self, trial):
        if self.config.sc_topk_range:
            self.config.sc_topk = self.config.sc_topk_range[trial.number]
        else:
            if self.config.alpha_range:
                self.config.alpha = self.config.alpha_range[trial.number]
            else:
                self.config.alpha = 1 - trial.number * 0.1
        self.config.save_path = Path(self.config.logdir) / f"a{self.config.alpha}_k{self.config.sc_topk}_{self.config.model}"
        self.config.save_path.mkdir(parents=True, exist_ok=True)
        with open(Path(self.config.save_path) / 'config.txt', 'w') as f:
            print(self.config, file=f)

    @time_desc_decorator('Build Graph')
    def build(self, trial=None):
        seed_torch(self.config.random_seed)
        if trial:
            self.optuna_build(trial)
        if self.config.threads_num:
            torch.set_num_threads(self.config.threads_num)
        if self.model is None or self.config.mode == 'optuna':
            self.model = model_dict[self.config.model](self.config)

        self.model.to(device)

        self.pat_meta_embed = nn.Embedding.from_pretrained(torch.from_numpy(load_embedding(self.config.meta_embed_path, 'np'))).to(device)
        self.pat_title_embed = nn.Embedding.from_pretrained(torch.from_numpy(load_embedding(self.config.pat_bert_title, 'np'))).to(device)
        self.pat_abstract_embed = nn.Embedding.from_pretrained(torch.from_numpy(load_embedding(self.config.pat_bert_abstract, 'np'))).to(device)            
        self.pat_claim_embed = nn.Embedding.from_pretrained(torch.from_numpy(load_embedding(self.config.pat_bert_claim, 'np'))).to(device)
        
        # Overview Parameters
        print('Model Parameters')
        for name, param in self.model.named_parameters():
            print(
                f"\t{name}\t{list(param.size())}\trequires_grad={param.requires_grad}"
            )

        if self.config.checkpoint:
            self.load_model(self.config.checkpoint, self.config.parallel)

        if self.config.mode in ['train', 'optuna', 'debug']:
            self.optimizer = optimizer_dict[self.config.optimizer](
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate
            )
            if self.config.parallel:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[self.config.local_rank]
                )

    def save_model(self, epoch):
        """Save parameters to checkpoint"""
        ckpt_path = os.path.join(self.config.save_path, f'{epoch}.pkl')
        print(f'Save parameters to {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def load_model(self, checkpoint, from_parallel=False):
        """Load parameters from checkpoint"""
        print(f'Load parameters from {checkpoint}')
        epoch = re.match(r"[0-9]*", os.path.basename(checkpoint)).group(0)
        self.epoch_i = int(epoch)
        model_state_dict = None

        # 如果使用torch.distributed训练保存的模型
        if from_parallel:
            state_dict = torch.load(checkpoint)
            # create new OrderedDict that does not contain `module.`
            from collections import OrderedDict
            model_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # remove `module.`
                name = k[7:]
                model_state_dict[name] = v
        else:
            model_state_dict = torch.load(checkpoint)
        self.model.load_state_dict(model_state_dict)

    def write_val_summary(self, epoch_i, res_dict):
        for k, v in res_dict.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        for k3, v3 in v2.items():
                            self.writer.add_scalar(f"{k}/{k2}/{k3}", v3, epoch_i + 1)
                    else:
                        self.writer.add_scalar(f"{k}/{k2}", v2, epoch_i + 1)
            else:
                self.writer.add_scalar(k, v, epoch_i + 1)

    @time_desc_decorator('Training Start!')
    def train(self, ):
        # 对不同进程loss取平均
        def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
            rt = tensor.clone()
            distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
            rt /= distributed.get_world_size()
            return rt

        if not self.config.parallel or self.config.local_rank == 0:
            self.writer = SummaryWriter(self.config.logdir)
        self.best_val_epoch = 0
        self.best_val_tgt = 0
        self.best_val_f1_epoch = 0
        self.best_val_f1 = 0
        epoch_loss_history = []
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = []
            self.model.train()

            for batch_i, (p_pla_idx, p_def_idx, p_pat_idx, n_pla_idx, n_def_idx, n_pat_idx, _) in enumerate(tqdm(self.train_data_loader)):
                p_pla_idx, p_def_idx, p_pat_idx = p_pla_idx.to(device), p_def_idx.to(device), p_pat_idx.to(device)
                n_pla_idx, n_def_idx, n_pat_idx = n_pla_idx.to(device), n_def_idx.to(device), n_pat_idx.to(device)
                p_pat_meta, n_pat_meta = self.pat_meta_embed(p_pat_idx), self.pat_meta_embed(n_pat_idx)
                p_pat_title, p_pat_abs, p_pat_claim = self.pat_title_embed(p_pat_idx), self.pat_abstract_embed(p_pat_idx), self.pat_claim_embed(p_pat_idx)
                n_pat_title, n_pat_abs, n_pat_claim = self.pat_title_embed(n_pat_idx), self.pat_abstract_embed(n_pat_idx), self.pat_claim_embed(n_pat_idx)
                loss_dict = self.model(p_pla_idx, p_def_idx, p_pat_meta, p_pat_title, p_pat_abs, p_pat_claim, n_pla_idx, n_def_idx, n_pat_meta, n_pat_title, n_pat_abs, n_pat_claim)
                # if training on multiple GPUs, store the loss of each GPU
                # loss = reduce_tensor(
                #     loss_dict['batch_loss'].data
                # ) if self.config.parallel else loss_dict['batch_loss']
                loss = loss_dict['loss']
                assert not isnan(loss.to(torch.device('cpu')).item())
                batch_loss_history.append(loss.item())
                if batch_i % self.config.print_every == 0 and (
                    not self.config.parallel or self.config.local_rank == 0
                ):
                    tqdm.write(
                        f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {loss.item():.3f}'
                    )
                    for loss_name, loss_value in loss_dict.items():
                        self.writer.add_scalar(
                            'training_'+loss_name,
                            loss_value.item(),
                            epoch_i * len(self.train_data_loader) + batch_i
                        )

                # reset gradient
                self.optimizer.zero_grad()
                # Back-propagation
                loss.backward()

                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                # Run optimizer
                self.optimizer.step()
                # self.scheduler.step()

            if not self.config.parallel or self.config.local_rank == 0:
                epoch_loss = np.sum(batch_loss_history
                                    ) / len(batch_loss_history)
                epoch_loss_history.append(epoch_loss)
                self.epoch_loss = epoch_loss
                print_str = f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f}'
                print(print_str)

                if self.val_data_loader:
                    print('\n<Validation>...')
                    self.eval(epoch_i)

                if self.test_data_loader and epoch_i >= self.config.begin_test_epoch and self.best_val_epoch == epoch_i + 1:
                    print('\n<Test>...')
                    self.eval(epoch_i, True)

                if epoch_i % self.config.save_every_epoch == 0 and self.best_val_epoch != epoch_i + 1:
                    self.save_model(epoch_i + 1)
                if self.config.early_stop and self.early_stop_epoch == self.config.early_stop:
                    print(f"early stop {self.early_stop_epoch} epochs")
                    break

        # if not self.config.parallel or self.config.local_rank == 0:
        #     self.save_model(self.config.n_epoch)
        print(f"best_val_epoch: {self.best_val_epoch}, best_val_auc: {self.best_val_tgt}")
        # print(f"best_val_f1_epoch: {self.best_val_f1_epoch}, best_val_f1: {self.best_val_f1}")
        with open(os.path.join(self.config.logdir, "best_val"), "w", encoding="utf-8") as f:
            print(f"best_val_epoch: {self.best_val_epoch}, best_val_auc: {self.best_val_tgt}", file=f)
        #     print(f"best_val_f1_epoch: {self.best_val_f1_epoch}, best_val_f1: {self.best_val_f1}", file=f)
        return epoch_loss_history

    def calc_loss(self, y, y_pred, label_atten=None):
        loss_dict = {}
        if self.config.loss_type == 'CTF':
            loss_dict['batch_loss'] = self.criterion(y, y_pred)
        
        else:
            raise NotImplementedError
        return loss_dict

    def criterion(self, y, y_pred):
        if self.config.loss_type in ['my_loss']:
            return F.nll_loss(y_pred, y.flatten())
        elif self.config.loss_type in ['label_smoothing']:
            return F.kl_div(y_pred, y, reduction='batchmean')
        else: 
            return F.cross_entropy(y_pred, y.flatten())

    def eval(self, epoch_i, test=False):
        self.model.eval()
        self.eval_loss = 0.0
        data_loader = self.test_data_loader if test else self.val_data_loader
        samples_0, samples_1, samples_2 = [], [], []
        preds_0, preds_1, preds_2 = [], [], []
        pos_details_0, pos_details_1, pos_details_2 = [], [], []
        neg_details_0, neg_details_1, neg_details_2 = [], [], []
        for batch_i, (p_pla_idx, p_def_idx, p_pat_idx, n_pla_idx, n_def_idx, n_pat_idx, neg_types) in enumerate(tqdm(data_loader)):
                with torch.no_grad():
                    sample = torch.cat([p_pla_idx.view(-1, 1), p_def_idx.view(-1, 1), p_pat_idx.view(-1, 1), n_pla_idx.view(-1, 1), n_def_idx.view(-1, 1), n_pat_idx.view(-1, 1)], dim=-1)
                    p_pla_idx, p_def_idx, p_pat_idx = p_pla_idx.to(device), p_def_idx.to(device), p_pat_idx.to(device)
                    n_pla_idx, n_def_idx, n_pat_idx = n_pla_idx.to(device), n_def_idx.to(device), n_pat_idx.to(device)
                    p_pat_meta, n_pat_meta = self.pat_meta_embed(p_pat_idx), self.pat_meta_embed(n_pat_idx)
                    p_pat_title, p_pat_abs, p_pat_claim = self.pat_title_embed(p_pat_idx), self.pat_abstract_embed(p_pat_idx), self.pat_claim_embed(p_pat_idx)
                    n_pat_title, n_pat_abs, n_pat_claim = self.pat_title_embed(n_pat_idx), self.pat_abstract_embed(n_pat_idx), self.pat_claim_embed(n_pat_idx)
                    loss_dict = self.model(p_pla_idx, p_def_idx, p_pat_meta, p_pat_title, p_pat_abs, p_pat_claim, n_pla_idx, n_def_idx, n_pat_meta, n_pat_title, n_pat_abs, n_pat_claim, True)
                    
                    self.eval_loss += loss_dict['loss'].item()
                    pred = self.model.res
                    pos_detail = self.model.pos_res
                    neg_detail = self.model.neg_res
                    # print(neg_types)
                    # print(neg_types==0)
                    samples_0 += sample[neg_types==0].tolist()
                    samples_1 += sample[neg_types==1].tolist()
                    samples_2 += sample[neg_types==2].tolist()
                    preds_0 += pred[neg_types==0].tolist()
                    preds_1 += pred[neg_types==1].tolist()
                    preds_2 += pred[neg_types==2].tolist()
                    pos_details_0 += pos_detail[neg_types==0].tolist()
                    pos_details_1 += pos_detail[neg_types==1].tolist()
                    pos_details_2 += pos_detail[neg_types==2].tolist()
                    neg_details_0 += neg_detail[neg_types==0].tolist()
                    neg_details_1 += neg_detail[neg_types==1].tolist()
                    neg_details_2 += neg_detail[neg_types==2].tolist()
            # print(labels)
        debug_dict = {
            'samples_0': samples_0,
            'samples_1': samples_1,
            'samples_2': samples_2,
            'preds_0': preds_0,
            'preds_1': preds_1,
            'preds_2': preds_2,
            'pos_details_0': pos_details_0,
            'pos_details_1': pos_details_1,
            'pos_details_2': pos_details_2,
            'neg_details_0': neg_details_0,
            'neg_details_1': neg_details_1,
            'neg_details_2': neg_details_2,
        }
            # print(labels)
        self.eval_loss /= len(data_loader)
        auc_0, prec_0, rec_0, f1_0, ndcg_0, mrr_0, hr_0 = leave_one_auc_scores_at_k(samples_0, preds_0, 0)
        auc_1, prec_1, rec_1, f1_1, ndcg_1, mrr_1, hr_1 = leave_one_auc_scores_at_k(samples_1, preds_1, 1)
        auc_2, prec_2, rec_2, f1_2, ndcg_2, mrr_2, hr_2 = leave_one_auc_scores_at_k(samples_2, preds_2, 2)

        avg_auc = (auc_0 + auc_1 + auc_2) / 3
        if test:
            self.write_eval_output(debug_dict)

        res_dict = {
                "eval loss": self.eval_loss,
                "auc_0": auc_0,
                "precision_0": prec_0,
                "recall_0": rec_0,
                "f1_0": f1_0,
                "ndcg_0": ndcg_0, 
                "mrr_0": mrr_0, 
                "hr_0": hr_0,
                "auc_1": auc_1,
                "precision_1": prec_1,
                "recall_1": rec_1,
                "f1_1": f1_1,
                "ndcg_1": ndcg_1, 
                "mrr_1": mrr_1, 
                "hr_1": hr_1,
                "auc_2": auc_2,
                "precision_2": prec_2,
                "recall_2": rec_2,
                "f1_2": f1_2,
                "ndcg_2": ndcg_2, 
                "mrr_2": mrr_2, 
                "hr_2": hr_2,
                "avg_auc": avg_auc,
            }
        if test:
            self.write_test_result(
                res_dict
            )
        else:
            print(f"epoch:{epoch_i + 1}")
            print(f"val_loss: {self.eval_loss}")
            if self.neg_type == 0:
                tgt = auc_0
            elif self.neg_type == 1:
                tgt = auc_1
            elif self.neg_type == 2:
                tgt = auc_2
            elif self.neg_type == 3:
                tgt = avg_auc
            print(f"AUC_{self.neg_type}: {tgt}")
            # print(
            #     f"precision: {avg_prec} recall: {avg} f1: {f1_0}\n"
            # )
            self.write_val_summary(epoch_i, res_dict)

            if tgt >= self.best_val_tgt:
                self.best_val_tgt, self.best_val_epoch = tgt, epoch_i + 1
                self.early_stop_epoch = 0
                self.save_model(epoch_i + 1)
            else:
                self.early_stop_epoch += 1

    def write_test_result(self, res_dict):
        epoch_i = self.epoch_i + 1 if self.config.mode in ['train', 'optuna'] else self.epoch_i
        with open(
            os.path.join(self.config.save_path, f"{epoch_i}_test_res"),
            "w",
            encoding="utf-8"
        ) as f:
            print(
                json.dumps(res_dict, ensure_ascii=False),
                file=f
            )

    def write_eval_output(self, res_dict):
        epoch_i = self.epoch_i + 1 if self.config.mode in ['train', 'optuna'] else self.epoch_i
        with open(
            os.path.join(self.config.save_path, f"{epoch_i}_test_output"),
            "w",
            encoding="utf-8"
        ) as f:
            print(
                json.dumps(res_dict, ensure_ascii=False),
                file=f
            )


    @time_desc_decorator('Training Start!')
    def optuna_train(self, trial):
        # average the loss of different processes
        def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
            rt = tensor.clone()
            distributed.all_reduce(rt, op=distributed.ReduceOp.SUM)
            rt /= distributed.get_world_size()
            return rt
        self.build(trial)
        if not self.config.parallel or self.config.local_rank == 0:
            self.writer = SummaryWriter(self.config.save_path)
        self.best_val_epoch = 0
        self.best_val_tgt = 0
        self.best_val_f1_epoch = 0
        self.best_val_f1 = 0
        epoch_loss_history = []
        self.epoch_i = 0
        for epoch_i in range(self.epoch_i, self.config.n_epoch):
            self.epoch_i = epoch_i
            batch_loss_history = []
            self.model.train()
            total_y, total_y_pred = [], []
            for batch_i, (seq, sc_labels, sc_probs, word_mask, y) in enumerate(tqdm(self.train_data_loader)):
                seq, sc_labels, sc_probs, y = seq.to(device), sc_labels.to(device), sc_probs.to(device), y.to(device)
                if self.config.enable_label_attention:
                    y_pred, sc_la = self.model(seq, None, sc_labels, word_mask, sc_probs)
                    loss_dict = self.calc_loss(y, y_pred, sc_la)
                else:
                    y_pred = self.model(seq, None, sc_labels, word_mask, sc_probs)
                    loss_dict = self.calc_loss(y, y_pred)
                # print(f"cross_entropy_loss: {cross_entropy_loss}, kl_div_loss_l: {kl_div_loss_l}, kl_div_loss_b: {kl_div_loss_b}")
                # if training on multiple GPUs, store the loss of each GPU
                loss = reduce_tensor(
                    loss_dict['batch_loss'].data
                ) if self.config.parallel else loss_dict['batch_loss']
                assert not isnan(loss.to(torch.device('cpu')).item())
                batch_loss_history.append(loss.item())
                total_y += y.flatten().tolist()
                total_y_pred += y_pred.max(-1,
                                           keepdim=True)[1].flatten().tolist()

                if batch_i % self.config.print_every == 0 and (
                    not self.config.parallel or self.config.local_rank == 0
                ):
                    tqdm.write(
                        f'Epoch: {epoch_i+1}, iter {batch_i}: loss = {loss.item():.3f}'
                    )
                    self.writer.add_scalar(
                        'train_loss',
                        loss.item(),
                        epoch_i * len(self.train_data_loader) + batch_i
                    )
                    for loss_name, loss_value in loss_dict.items():
                        if loss_name == 'batch_loss':
                            continue
                        self.writer.add_scalar(
                            loss_name,
                            loss_value.item(),
                            epoch_i * len(self.train_data_loader) + batch_i
                        )

                # reset gradient
                self.optimizer.zero_grad()
                # Back-propagation
                loss.backward()

                # Gradient cliping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)

                # Run optimizer
                self.optimizer.step()
                # self.scheduler.step()

            if not self.config.parallel or self.config.local_rank == 0:
                epoch_loss = np.sum(batch_loss_history
                                    ) / len(batch_loss_history)
                epoch_loss_history.append(epoch_loss)
                self.epoch_loss = epoch_loss
                train_acc = accuracy_score(total_y, total_y_pred)
                print_str = f'Epoch {epoch_i+1} loss average: {epoch_loss:.3f} train accuracy: {train_acc:.3f}'
                print(print_str)
                self.writer.add_scalar('train_acc', train_acc, epoch_i + 1)
                if epoch_i % self.config.save_every_epoch == 0:
                    self.save_model(epoch_i + 1)

                if self.val_data_loader:
                    print('\n<Validation>...')
                    self.eval_percentage(epoch_i)

                if self.test_data_loader and epoch_i >= self.config.begin_test_epoch:
                    print('\n<Test>...')
                    self.eval_percentage(epoch_i, True, self.config.get_test_text)
                if self.config.early_stop and self.early_stop_epoch == self.config.early_stop:
                    print(f"early stop {self.early_stop_epoch} epochs")
                    break
        if not self.config.parallel or self.config.local_rank == 0:
            self.save_model(self.config.n_epoch)
        print(f"best_val_epoch: {self.best_val_epoch}, best_val_accuracy: {self.best_val_tgt}")
        print(f"best_val_f1_epoch: {self.best_val_f1_epoch}, best_val_f1: {self.best_val_f1}")
        with open(os.path.join(self.config.save_path, "best_val"), "w", encoding="utf-8") as f:
            print(f"best_val_epoch: {self.best_val_epoch}, best_val_accuracy: {self.best_val_tgt}", file=f)
            print(f"best_val_f1_epoch: {self.best_val_f1_epoch}, best_val_f1: {self.best_val_f1}", file=f)
        val_target = None
        if self.config.val_target == 'acc':
            val_target = self.best_val_tgt
        elif self.config.val_target == 'f1':
            val_target = self.best_val_f1
        return val_target

    def encode_comp(self, indices):
        self.model.eval()
        indices = torch.tensor(indices, dtype=torch.long).to(device)
        with torch.no_grad():
            return self.model.encode_comp(indices)

    def encode_pat(self, indices):
        self.model.eval()
        indices = torch.tensor(indices, dtype=torch.long).to(device)
        pat_meta = self.pat_meta_embed(indices)
        pat_title, pat_abs, pat_claim = self.pat_title_embed(indices), self.pat_abstract_embed(indices), self.pat_claim_embed(indices)
        with torch.no_grad():
            return self.model.encode_pat(pat_meta, pat_title, pat_abs, pat_claim)


    @time_desc_decorator('debugging Start!')
    def debug(self, ):
        return self.train()