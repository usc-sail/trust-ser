import thop
import json
import yaml
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import argparse, logging
import torch.multiprocessing
from torchscan import summary
import copy, time, pickle, shutil, sys, os, pdb

from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import defaultdict, deque
from ptflops import get_model_complexity_info
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau


sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[2])))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[2]), 'model'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[2]), 'experiment'))
sys.path.append(os.path.join(str(Path(os.path.realpath(__file__)).parents[2]), 'dataloader'))

from utils import parse_finetune_args, set_seed, log_epoch_result, log_best_result

# from utils
from evaluation import EvalMetric
from downstream_models import DNNClassifier, CNNSelfAttention
from dataloader import load_finetune_audios, set_finetune_dataloader, return_speakers
from pretrained_backbones import Wav2Vec, APC, TERA, WavLM, WhisperTiny, WhisperBase, WhisperSmall

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Model basic information
# Model hidden states information
hid_dim_dict = {
    "wav2vec2_0":       768,
    "tera":             768,
    "wavlm":            768,
    "whisper_small":    768,
    "whisper_base":     512,
    "whisper_tiny":     384,
    "apc":              512,
}

# Model number of encoding layers
num_enc_layers_dict = {
    "wav2vec2_0":       12,
    "wavlm":            12,
    "whisper_small":    12,
    "whisper_base":     6,
    "tera":             4,
    "whisper_tiny":     4,
    "apc":              3,
}


if __name__ == '__main__':

    # Argument parser
    args = parse_finetune_args()

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    
    # Define the model wrapper
    if args.pretrain_model == "wav2vec2_0":
        # Wav2vec2_0 Wrapper
        backbone_model = Wav2Vec().to(device)
    elif args.pretrain_model == "apc":
        # APC wrapper from superb
        backbone_model = APC().to(device)
    elif args.pretrain_model == "tera":
        # TERA wrapper from superb
        backbone_model = TERA().to(device)
    elif args.pretrain_model == "wavlm":
        # Wavlm wrapper from huggingface
        backbone_model = WavLM().to(device)
    elif args.pretrain_model == "whisper_tiny":
        # Whisper tiny wrapper from huggingface
        backbone_model = WhisperTiny().to(device)
    elif args.pretrain_model == "whisper_base":
        # Whisper base wrapper from huggingface
        backbone_model = WhisperBase().to(device)
    elif args.pretrain_model == "whisper_small":
        # Whisper small wrapper from huggingface
        backbone_model = WhisperSmall().to(device)

    # Read trainable params
    model_parameters = filter(lambda p: p.requires_grad, backbone_model.backbone_model.encoder.parameters())
    params = sum([np.prod(p.size()) for p in backbone_model.backbone_model.encoder.parameters()])
    # params = sum([np.prod(p.size()) for p in backbone_model.backbone_model.parameters()])
    # print(backbone_model.backbone_model.encoder)
    # pdb.set_trace()
    logging.info(f'Backbone Model {args.pretrain_model} Trainable params size: {params/1e6:.2f} M')
    