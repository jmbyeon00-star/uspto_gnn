from module import get_config
import os
import numpy as np
import optuna
import logging
from module import seed_torch, MANTFDataset, collate
from pathlib import Path
from solver import Solver
from numpy import set_printoptions
set_printoptions(threshold=np.inf, linewidth=np.nan)

from torch.utils.data import DataLoader

if __name__ == '__main__':
    config = get_config()
    print(config)
    with open(os.path.join(config.save_path, 'config.txt'), 'w') as f:
        print(config, file=f)

    seed_torch(config.random_seed)

    TrainDataset = MANTFDataset
    EvalDataset = MANTFDataset

    collate_train = collate
    collate_eval = collate

    train_loader, val_loader = None, None
    if config.mode == 'train':
        train_data = TrainDataset(
            config.patent_idx_path, 
            config.company_idx_path, 
            config.train_file, 
            config.neg_types
            )
        val_data = EvalDataset(
            config.patent_idx_path, 
            config.company_idx_path, 
            config.dev_file, 
            [0, 1, 2], 
            False
            )
        train_loader = DataLoader(train_data, shuffle=True, batch_size=config.batch_size, collate_fn=collate_train)
        val_loader = DataLoader(val_data, shuffle=False, batch_size=config.batch_size, collate_fn=collate_eval)
        
    test_loader = None
    if config.test_file:
        test_data = EvalDataset(
            config.patent_idx_path, 
            config.company_idx_path, 
            config.test_file, 
            [0, 1, 2], 
            False
            )
        test_loader = DataLoader(test_data, shuffle=False, batch_size=config.batch_size, collate_fn=collate_eval)
    
    solver = Solver(config, train_loader, val_loader, test_loader)

    if config.mode == "train":
        solver.build()
        solver.train()
    elif config.mode == "debug":
        solver.build()
        solver.debug()
    elif config.mode == 'test':
        solver.build()
        solver.eval(None, True)
    elif config.mode == "optuna":
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)  # Setup the root logger.
        logger.addHandler(logging.FileHandler(Path(config.logdir) / "optuna.log", mode="w"))

        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

        study = optuna.create_study(direction="maximize")
        study.optimize(solver.optuna_train, n_trials=config.n_trials)

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
        with open(os.path.join(config.logdir, "best_trial"), "w", encoding="utf-8") as f:
            for key, value in trial.params.items():
                f.write("    {}: {}\n".format(key, value))