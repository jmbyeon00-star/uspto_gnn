import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class GCN_relu(nn.Module):
    def __init__(self, in_feat_size, out_feat_size, gcn_layers=1, dropout=0.5):
        super(GCN_relu, self).__init__()

        """
        plaintiff_size: number of plaintiff companies
        defendant_size: number of defendant companies
        patent_size: number of patents including litigated and non-litigated patents
        factor_size: dimension of encoded company and patent embeddings
        dropout: dropout rate of NCNN model
        """
        self.in_feat_size = in_feat_size
        self.out_feat_size = out_feat_size
        # more cnn layers can be adapted, which depends on the users' design
        # here is just a demo
        self.conv_1 = GraphConv(in_feat_size, out_feat_size)
        self.dropout = nn.Dropout(dropout)
        self.convs = nn.ModuleList([GraphConv(out_feat_size, out_feat_size) for _ in range(gcn_layers-1)])

    def forward(self, g, feat):
        """
        :param p_text: litigated patent text tensor including 1 title, 1 abstract, 30 claims, totally 32 slices
                       shape=[batch_size, slice_num=32, word_number_each_slice=300]
        :param n_text: non-litigated patent text tensor, similar to p_text
        :param p_meta: litigated patent meta tensor, which is pretrained by an attribute citation network
                       shape=[batch_size, 1, 1, meta_embedding_size=100]
        :param n_meta: non-litigated patent meta tensor, similar to p_meta
        :return: loss, harmonic parameters of should adjust to real data
        """
        h = self.dropout(F.relu(self.conv_1(g, feat)))
        for conv in self.convs:
            h = self.dropout(F.relu(conv(g, h)))
        return h