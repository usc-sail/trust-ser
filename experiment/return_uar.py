import json
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import defaultdict, deque
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[1]), 'dataloader'))

from utils import parse_finetune_args, set_seed, log_epoch_result, log_best_result

# from utils
from evaluation import EvalMetric
from pretrained_backbones import Wav2Vec, APC, TERA
from downstream_models import DNNClassifier, CNNSelfAttention
from dataloader import load_finetune_audios, set_finetune_dataloader, return_speakers

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# model basic information
hid_dim_dict = {
    "wav2vec2_0":   768,
    "tera":         768,
    "apc":          512,
}

num_enc_layers_dict = {
    "wav2vec2_0":   12,
    "tera":         4,
    "apc":          3,
}

if __name__ == '__main__':

    # argument parser
    args = parse_finetune_args()

    # number of folds in the exp
    if args.dataset == "msp-improv": total_folds = 7
    else: total_folds = 6
    
    uar_df = pd.DataFrame(
        index=[epoch for epoch in range(args.num_epochs)], 
        columns=[fold_idx for fold_idx in range(1, total_folds)]
    )
    for fold_idx in range(1, total_folds):
        # Log/model dir
        log_dir = Path(args.log_dir).joinpath(
            args.dataset, 
            args.pretrain_model,
            f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.downstream_model}_conv{args.conv_layers}_hid{args.hidden_size}_{args.pooling}'
        )
        
        with open(str(Path(log_dir).joinpath(f'fold_{fold_idx}.json')), "r") as f:
            split_dict = json.load(f)
        
        for epoch in range(args.num_epochs):
            uar_df.loc[epoch, fold_idx] = split_dict[f"{epoch}"]["test"]["uar"]
    
    uar_dir = Path(args.uar_dir).joinpath(
        args.dataset, 
        args.pretrain_model,
        f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.downstream_model}_conv{args.conv_layers}_hid{args.hidden_size}_{args.pooling}'
    )
    Path.mkdir(uar_dir, parents=True, exist_ok=True)
    uar_df.to_csv(str(uar_dir.joinpath("uar.csv")))
    