from pathlib import Path

BASE = Path('/home2/jmbyeon/developments/workspace/apps/uspto_litigation/소송예측모델 개발')
EXP_DIR = Path(__file__).resolve().parent
SNAP_DIR = BASE / '소송예측9월/0916PSEUDO_ONESHOT/snapshots_clean_0917'
DATA_GNN = BASE / 'data_gnn'
OWNERSHIP_PATH = BASE / 'data/with_counter/ownership_chain_v2.parquet'
USABLE_PATH = BASE / 'data/with_counter/gnn_usable_cases_v2.parquet'
MATCH_DIR = BASE / '소송예측9월/0916PSEUDO_ONESHOT/pseudo_matching/results'

SEED = 42
HIDDEN = 128
DROPOUT = 0.30
EPOCHS = 1000
PATIENCE = 60
VAL_EVERY = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

TRAIN_RATIO = 0.70
N_HARD_TRAIN = 2
N_RANDOM_TRAIN = 3
N_HARD_VAL = 400
N_TOTAL_VAL = 1000
PSEUDO_POOL_MIN_SCORE = 75.0
RANK_CHUNK_SIZE = 100000

TRAIN_FOLDS = [(2004,2005,2006),(2006,2007,2008),(2008,2009,2010),
               (2010,2011,2012),(2012,2013,2014),(2014,2015,2016),
               (2016,2017,2018)]
TEST_FOLD = (2018,2019,2020)
