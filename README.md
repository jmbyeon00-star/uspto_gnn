## MANTF-master

This repository is the implementation of the paper: Han Wu, Guanqi Zhu, Qi Liu, Hengshu Zhu, Hao Wang, Hongke Zhao, Chuanren Liu, Enhong Chen, Hui Xiong. A Multi-aspect Neural Tensor Factorization Framework for Patent Litigation Prediction.

## Installation

Install pytorch 1.13.1

Install dgl-cu110 0.6.1

Install scikit-learn 1.0.2

## Dataset

Download the dataset from 链接：https://rec.ustc.edu.cn/share/8ae7f390-0697-11ee-9a61-e37520c2839a密码：r9z8

unzip data.zip

## Running the code

mkdir pt

CUDA_VISIBLE_DEVICES=3 python train.py \
    --mode=train \
    --threads_num=6 \
    --n_epoch=1000 \
    --batch_size=128 \
    --random_seed=2021 \
    --learning_rate=1e-4 \
    --optimizer=Adam \
    --model=MANTF \
    --company_num=19772 \
    --patent_num=89552 \
    --factor_size=300 \
    --dropout=0.5 \
    --patent_idx_path=data/patex/samples/patex_patent_index.txt \
    --company_idx_path=data/patex/samples/patex_company_index.txt \
    --comp_graph_path=data/patex/features/patex_percentage_wise_80_l_d_10_graph.npy \
    --comp_feat_path=data/patex/features/patex_comp_feat.npy \
    --comp_feat_freeze=n \
    --bert_size=768 \
    --gcn_dropout=0.2 \
    --pat_bert_title=data/patex/features/patex_pat_titles.npy \
    --pat_bert_abstract=data/patex/features/patex_pat_abstracts.npy \
    --pat_bert_claim=data/patex/features/patex_pat_avg_claims.npy \
    --meta_embed_path=data/patex/features/patex_patent_deepwalk_embedding.npy \
    --train_file=data/patex/samples/patex_percentage_wise_train_80 \
    --dev_file=data/patex/samples/patex_percentage_wise_val \
    --test_file=data/patex/samples/patex_percentage_wise_test \
    --save_every_epoch=1 \
    --begin_test_epoch=0 \
    --early_stop=10 \
    --neg_types 0 1 2 \
    --description='wuhan_patex_real_new2_loss012345_300'\
    --r_var=0.000001 \
    --p_var=0.01 \
    --u_var=0.01 \
    --v_var=0.01 \
    --g_var=0.01 \
    --q_var=0.01 \
    --lambda0=0.001 \
   
   
CUDA_VISIBLE_DEVICES=1 python train.py \
    --mode=train \
    --threads_num=6 \
    --n_epoch=1000 \
    --batch_size=128 \
    --random_seed=10 \
    --learning_rate=1e-4 \
    --optimizer=Adam \
    --model=MANTF \
    --company_num=35066 \
    --patent_num=257958 \
    --factor_size=500 \
    --dropout=0.5 \
    --patent_idx_path=data/stanford/samples/stanford_patent_index.txt \
    --company_idx_path=data/stanford/samples/stanford_company_index.txt \
    --comp_graph_path=data/stanford/features/stanford_percentage_wise_80_l_d_10_graph.npy \
    --comp_feat_path=data/stanford/features/stanford_comp_feat.npy \
    --comp_feat_freeze=n \
    --bert_size=768 \
    --gcn_dropout=0.2 \
    --pat_bert_title=data/stanford/features/stanford_pat_titles.npy \
    --pat_bert_abstract=data/stanford/features/stanford_pat_abstracts.npy \
    --pat_bert_claim=data/stanford/features/stanford_pat_avg_claims.npy \
    --meta_embed_path=data/stanford/features/stanford_patent_deepwalk_embedding.npy \
    --train_file=data/stanford/samples/stanford_percentage_wise_train_80 \
    --dev_file=data/stanford/samples/stanford_percentage_wise_val \
    --test_file=data/stanford/samples/stanford_percentage_wise_test \
    --save_every_epoch=1 \
    --begin_test_epoch=0 \
    --early_stop=10 \
    --neg_types 0 1 2 \
    --description='wuhan_stanford_real_loss012345_500_seed10'\
    --r_var=0.0001 \
    --p_var=0.01 \
    --u_var=0.01 \
    --v_var=0.01 \
    --g_var=0.01 \
    --q_var=0.01 \
    --lambda0=0.001 \
  
