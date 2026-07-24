import numpy as np
from .layers import GCN_relu
from .config import device
from .utils import load_embedding
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class MANTF(nn.Module):
    def __init__(self, config):
        super(MANTF, self).__init__()

        """
        plaintiff_size: number of plaintiff companies
        defendant_size: number of defendant companies
        patent_size: number of patents including litigated and non-litigated patents
        factor_size: dimension of encoded company and patent embeddings
        word_emb_size: dimension of input word embdding of patents
        dropout: dropout rate of NCNN model
        """

        self._r_var = config.r_var
        self._u_var = config.u_var
        self._v_var = config.v_var
        self._p_var = config.p_var
        self._g_var = config.g_var
        self._q_var = config.q_var

        self._lambda1 = 0.5 * (self._r_var / self._u_var) # real new
        self._lambda2 = 0.5 * (self._r_var / self._v_var)
        self._lambda3 = 0.5 * (self._r_var / self._p_var)
        self._lambda4 = 0.5 * (self._r_var / self._g_var)
        self._lambda5 = 0.5 * (self._r_var / self._q_var)

        self._lambda0 = config.lambda0

        self._dropout = nn.Dropout(config.dropout)
        self._comp_g = self.build_graph(config.comp_graph_path, config.company_num).to(device)
        if config.comp_feat_path:
            self._comp_feat = nn.Parameter(
                torch.tensor(load_embedding(config.comp_feat_path, 'np'), dtype=torch.float32), 
                requires_grad=not config.comp_feat_freeze
                )
        else:
            self._comp_feat = nn.Parameter(
                torch.randn(config.company_num, config.factor_size), 
                requires_grad=True
                )
        comp_feat_size = self._comp_feat.shape[-1]
        self._gcn = GCN_relu(comp_feat_size, config.factor_size, config.gcn_layers, config.gcn_dropout)
        self._comp_mask_pla = nn.Parameter(5**0.5 * torch.randn(1, config.factor_size), requires_grad=True)
        self._comp_mask_def = nn.Parameter(5**0.5 * torch.randn(1, config.factor_size), requires_grad=True)

        # MLP
        self._fc_title = nn.Linear(config.bert_size, config.bert_size)
        self._fc_abstract = nn.Linear(config.bert_size, config.bert_size)
        self._fc_claim = nn.Linear(config.bert_size, config.bert_size)
        self._fc_text = nn.Linear(3*config.bert_size, 3*config.bert_size)
        self._mlp = nn.Linear(3*config.bert_size+100, config.factor_size)
        
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self._fc_title.weight, gain=gain)
        nn.init.zeros_(self._fc_title.bias)
        nn.init.xavier_normal_(self._fc_abstract.weight, gain=gain)
        nn.init.zeros_(self._fc_abstract.bias)
        nn.init.xavier_normal_(self._fc_claim.weight, gain=gain)
        nn.init.zeros_(self._fc_claim.bias)
        nn.init.xavier_normal_(self._fc_text.weight, gain=gain)
        nn.init.zeros_(self._fc_text.bias)
        nn.init.xavier_normal_(self._mlp.weight, gain=gain)
        nn.init.zeros_(self._mlp.bias)
 
    def forward(self, p_pla_idx, p_def_idx, p_pat_meta, p_pat_title, p_pat_abs, p_pat_claim, n_pla_idx, n_def_idx, n_pat_meta, n_pat_title, n_pat_abs, n_pat_claim, eval_mode=False):
        # company embedding
        comp_mask_pla = torch.sigmoid(self._comp_mask_pla)
        comp_mask_def = torch.sigmoid(self._comp_mask_def)
        comp_feat = self._gcn(self._comp_g, self._comp_feat)
        pla_feat = comp_feat * comp_mask_pla
        def_feat = comp_feat * comp_mask_def

        p_pla_factor_emb = pla_feat[p_pla_idx]
        p_def_factor_emb = def_feat[p_def_idx]
        n_pla_factor_emb = pla_feat[n_pla_idx]
        n_def_factor_emb = def_feat[n_def_idx]

        # add variance for plaintiffs and defendants
        p_pla_emb = p_pla_factor_emb + (self._u_var**0.5) * torch.randn(*p_pla_factor_emb.shape).to(device)
        p_def_emb = p_def_factor_emb + (self._v_var**0.5) * torch.randn(*p_def_factor_emb.shape).to(device)
        n_pla_emb = n_pla_factor_emb + (self._u_var**0.5) * torch.randn(*n_pla_factor_emb.shape).to(device)
        n_def_emb = n_def_factor_emb + (self._v_var**0.5) * torch.randn(*n_def_factor_emb.shape).to(device)

        # patent embedding
        p_pat_title = self._fc_title(p_pat_title)
        p_pat_abs = self._fc_abstract(p_pat_abs)
        p_pat_claim = self._fc_claim(p_pat_claim)
        p_text_repr = self._fc_text(torch.cat((p_pat_title, p_pat_abs, p_pat_claim), dim=-1))

        n_pat_title = self._fc_title(n_pat_title)
        n_pat_abs = self._fc_abstract(n_pat_abs)
        n_pat_claim = self._fc_claim(n_pat_claim)
        n_text_repr = self._fc_text(torch.cat((n_pat_title, n_pat_abs, n_pat_claim), dim=-1))
        
        p_contact = torch.cat((p_text_repr, p_pat_meta), -1)
        n_contact = torch.cat((n_text_repr, n_pat_meta), -1)

        p_pat_factor_emb = self._mlp(p_contact)
        n_pat_factor_emb = self._mlp(n_contact)

        # add variance for patents
        p_pat_emb = p_pat_factor_emb + (self._p_var**0.5) * torch.randn(*p_pat_factor_emb.shape).to(device)
        n_pat_emb = n_pat_factor_emb + (self._p_var**0.5) * torch.randn(*n_pat_factor_emb.shape).to(device)

        # Prediction
        p_res1 = torch.sum(p_pla_emb * p_def_emb, dim=-1)
        p_res2 = torch.sum(p_pla_emb * p_pat_emb, dim=-1)
        p_res3 = torch.sum(p_pat_emb * p_def_emb, dim=-1)
        p_res = p_res1 + p_res2 + p_res3
        p_res += self._r_var**0.5 * torch.randn(*p_res.shape).to(device)

        n_res1 = torch.sum(n_pla_emb * n_def_emb, dim=-1)
        n_res2 = torch.sum(n_pla_emb * n_pat_emb, dim=-1)
        n_res3 = torch.sum(n_pat_emb * n_def_emb, dim=-1)
        n_res = n_res1 + n_res2 + n_res3
        n_res += self._r_var**0.5 * torch.randn(*n_res.shape).to(device)

        if eval_mode:
            self.res = torch.cat((torch.sigmoid(p_res).view(-1, 1), torch.sigmoid(n_res).view(-1, 1)), dim=-1)
            self.pos_res = torch.cat((p_res1.view(-1, 1), p_res2.view(-1, 1), p_res3.view(-1, 1)), dim=-1)
            self.neg_res = torch.cat((n_res1.view(-1, 1), n_res2.view(-1, 1), n_res3.view(-1, 1)), dim=-1)

        # Loss
        loss = torch.mean(-F.logsigmoid(p_res - n_res)) 
        loss0 = self.emb_norm([p_pla_emb, n_pla_emb, p_def_emb, n_def_emb, p_pat_emb, n_pat_emb])
        loss1 = self.get_subnorm(p_pla_factor_emb, p_pla_emb) + self.get_subnorm(n_pla_factor_emb, n_pla_emb)
        loss2 = self.get_subnorm(p_def_factor_emb, p_def_emb) + self.get_subnorm(n_def_factor_emb, n_def_emb)
        loss3 = self.get_subnorm(p_pat_factor_emb, p_pat_emb) + self.get_subnorm(n_pat_factor_emb, n_pat_emb)
        loss4 = self.pen_norm()
        loss5 = self.mcen_norm()

        loss += self._lambda0 * loss0
        loss += self._lambda1 * loss1
        loss += self._lambda2 * loss2
        loss += self._lambda3 * loss3
        loss += self._lambda4 * loss4
        loss += self._lambda5 * loss5

        return {'loss': loss, 'loss0': loss0, 'loss1': loss1, 'loss2': loss2, 'loss3': loss3, 'loss4': loss4, 'loss5': loss5}

    def build_graph(self, graph_path, num_nodes):
        edges = np.load(graph_path)
        g = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=num_nodes, idtype=torch.int32)
        g = dgl.to_simple(g)
        g = dgl.add_self_loop(dgl.to_bidirected(g))
        return g
    
    def emb_norm(self, embs):
        loss = 0.
        for emb in embs:
            loss += torch.mean(torch.norm(emb))
        return loss

    def pen_norm(self):
        loss = 0.
        for item in [self._fc_title, self._fc_abstract, self._fc_claim, self._fc_text, self._mlp]:
            loss += torch.mean(torch.norm(item.weight))
        return loss
    
    def mcen_norm(self):
        return torch.mean(torch.norm(self._gcn.conv_1.weight))

    def get_subnorm(self, emb_1, emb_2):
        return torch.mean(torch.norm(torch.sub(emb_1, emb_2)))
