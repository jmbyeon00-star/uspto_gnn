import os
import argparse
from datetime import datetime
import pprint
import torch
from torch import optim
import torch.nn as nn

debug = False
device = torch.device('cuda' if torch.cuda.is_available() and not debug else 'cpu')

PAD_IDX, UNK_IDX = 0, 1
PAD = '<pad>'
UNK = '<unk>'

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
optimizer_dict = {'RMSprop': optim.RMSprop,
                  'Adam': optim.Adam,
                  'Adagrad': optim.Adagrad,
                  'AdamW': 'AdamW',
                  'sgd': optim.SGD}
save_dir = os.path.join(project_dir, './pt/')
pred_dir = os.path.join(project_dir, './pred/')

def str2bool(v):
    """string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config(object):
    def __init__(self, **kwargs):
        """Configuration Class: set kwargs as class attributes with setattr"""
        if kwargs is not None:
            for key, value in kwargs.items():
                setattr(self, key, value)
        if getattr(self, 'text_embed_path', None) and 'glove' in self.text_embed_path:
            self.text_embed_size = 300
        # Pickled Dataframes
        os.makedirs(pred_dir, exist_ok=True)
        self.pred_path = os.path.join(pred_dir, 'res.txt')
        # Save path
        dir_name = self.model
        if self.mode in ['train', 'debug'] and self.checkpoint is None:
            time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.save_path = os.path.join(save_dir, f"{self.description}_{dir_name}_{time_now}")
            self.logdir = self.save_path
            os.makedirs(self.save_path, exist_ok=True)
        if self.mode == 'optuna' and self.checkpoint is None:
            if self.alpha_range:
                self.n_trials = len(self.alpha_range)
            time_now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            self.save_path = os.path.join(save_dir, f"optuna_{self.description}_{dir_name}_{time_now}")
            self.logdir = self.save_path
            os.makedirs(self.save_path, exist_ok=True)
        elif self.checkpoint is not None:
            assert os.path.exists(self.checkpoint)
            self.save_path = os.path.dirname(self.checkpoint)
            self.logdir = self.save_path

    def __str__(self):
        """Pretty-print configurations in alphabetical order"""
        # config_str = 'Configurations\n'
        config_str = pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """
    Get configurations as attributes of class
    1. Parse configurations with argparse.
    2. Create Config class initilized with parsed kwargs.
    3. Return Config class.
    """
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument('--mode', type=str, default='train')

    # Train
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epoch', type=int, default=1000)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--checkpoint', type=str, default=None)

    # Model
    parser.add_argument('--model', type=str, default='SimpleHGNPred')
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--company_num', type=int, default=19772)
    parser.add_argument('--patent_num', type=int, default=89552)
    parser.add_argument('--factor_size', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)

    parser.add_argument('--r_var', type=float, default=0.01)
    parser.add_argument('--p_var', type=float, default=0.01)
    parser.add_argument('--u_var', type=float, default=0.01)
    parser.add_argument('--v_var', type=float, default=0.01)
    parser.add_argument('--g_var', type=float, default=0.01)
    parser.add_argument('--q_var', type=float, default=0.01)

    parser.add_argument('--lambda0', type=float, default=0.001)
    parser.add_argument('--lambda1', type=float, default=0.001)
    parser.add_argument('--lambda2', type=float, default=0.001)
    parser.add_argument('--lambda3', type=float, default=0.001)
    parser.add_argument('--lambda4', type=float, default=0.001)
    parser.add_argument('--lambda5', type=float, default=0.001)
    
    parser.add_argument('--caculate', type=str2bool, default=False)
    
    # 负采样的种类
    parser.add_argument('--neg_types', nargs='+', type=int, default=[0, 1, 2])

    # GCN
    parser.add_argument('--gcn_layers', type=int, default=2)
    parser.add_argument('--gcn_dropout', type=float, default=0.2)

    # BERT
    parser.add_argument('--bert_size', type=int, default=768)
    # patent
    parser.add_argument('--patent_idx_path', type=str, default='data/patex/samples/patex_patent_index.txt')
    parser.add_argument('--pat_bert_path', type=str, default=None, help='将title,abstract,claims全部avg后的embed')
    parser.add_argument('--pat_bert_freeze', type=str2bool, default=False)
    #
    parser.add_argument('--pat_bert_title', type=str, default='data/patex/features/patex_pat_titles.npy')
    parser.add_argument('--pat_bert_abstract', type=str, default='data/patex/features/patex_pat_abstracts.npy')
    parser.add_argument('--pat_bert_claim', type=str, default='data/patex/features/patex_pat_avg_claims.npy')

    # Company
    parser.add_argument('--company_idx_path', type=str, default='data/patex/samples/patex_company_index.txt')
    parser.add_argument('--comp_graph_path', type=str, default='data/patex/features/patex_percentage_wise_20_l_d_10_graph.npy')
    parser.add_argument('--comp_feat_path', type=str, default='data/patex/features/patex_comp_feat.npy')
    parser.add_argument('--comp_feat_freeze', type=str2bool, default=True)

    # embedding
    # parser.add_argument('--text_embed_path', type=str, default=None)
    # parser.add_argument('--text_embed_token_path', type=str, default=None)
    parser.add_argument('--meta_embed_path', type=str, default='data/patex/features/patex_patent_deepwalk_embedding.npy')
    parser.add_argument('--text_embed_freeze', type=str2bool, default=True)
    parser.add_argument('--meta_embed_freeze', type=str2bool, default=True)

    # 那baseline
    parser.add_argument('--baseline_graph_path', type=str, default='data/patex/samples/ui_graph.npy')
    parser.add_argument('--baseline_hete_graph_path', type=str, default='data/patex/samples/pla_def_pat_graph.npy')

    # NCFTF parameters
    parser.add_argument('--NCF_model', type=str, default='MLP') #'MLP', 'GMF', 'NeuMF-end'

    parser.add_argument('--begin_test_epoch', type=int, default=0)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--plot_every_epoch', type=int, default=1)
    parser.add_argument('--save_every_epoch', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=None)
    parser.add_argument('--load_train_data', type=str2bool, default=False)
    parser.add_argument('--parallel', type=str2bool, default=False)
    parser.add_argument('--threads_num', type=int, default=6)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--description', type=str, default='simplehgn_debug')

    # Optuna
    parser.add_argument('--n_trials', type=int, default=150)
    parser.add_argument('--val_target', type=str, default='auc')
    parser.add_argument('--alpha_range', nargs='+', type=float, default=None)

    # Data
    parser.add_argument('--train_file', type=str, default='data/patex/samples/patex_percentage_wise_train_80')
    parser.add_argument('--dev_file', type=str, default='data/patex/samples/patex_percentage_wise_val')
    parser.add_argument('--test_file', type=str, default='data/patex/samples/patex_percentage_wise_test')
    # parser.add_argument('--patent_texts_file', type=str, default=None)
    # DataParallel
    parser.add_argument('--local_rank', type=int, default=-1)

    # Parse arguments
    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    # Namespace => Dictionary
    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)

if __name__ == "__main__":
    print(project_dir)