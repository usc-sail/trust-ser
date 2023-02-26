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
from dataloader import load_finetune_audios, return_dataset_stats, return_speakers
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

def count_glops(module, input, output):
    if isinstance(module, torch.nn.Embedding):
        return input[0].size(0) * input[1].size(1) * module.embedding_dim / 1e9
    else:
        n_flops = thop.count_ops(module, input, output)
        return n_flops / 1e9

def validate_epoch(
    model, 
    device,
    split:  str="Validation"
):
    # Set to eval mode
    model.eval()
    backbone_model.eval()
    
    # Save eval metrics
    eval_metric = EvalMetric()

    with torch.no_grad():
        input_size = (1, 16000*6)  # Audio waveform with 1 channel and 16000 samples
        input_data = torch.randn(input_size).cuda()
        if "whisper" in args.pretrain_model:
            
            input_features = backbone_model.feature_extractor(
                input_data[0].detach().cpu(), 
                return_tensors="pt", 
                sampling_rate=16000,
                max_length=len(input_data[0])
            )
            input_features = input_features.input_features.cuda()
            
            # Return length
            length = backbone_model.get_feat_extract_output_lengths(len(input_data[0]))
            
            # Replace positional embeddings
            backbone_model.backbone_model.encoder.embed_positions = backbone_model.backbone_model.encoder.embed_positions.from_pretrained(backbone_model.embed_positions[:length])
            shape = backbone_model.backbone_model.encoder.embed_positions.weight.shape
            backbone_model.backbone_model.encoder.embed_positions = nn.Linear(shape[1], shape[0], bias=False).to(device)
            flops, params = thop.profile(backbone_model.backbone_model.encoder, inputs=(input_features,))
        elif args.pretrain_model in ["wav2vec2_0", "wavlm"]:
            flops, params = thop.profile(backbone_model, inputs=(input_data,))
        elif args.pretrain_model in ["tera", "apc"]:
            flops, params = thop.profile(backbone_model, inputs=(input_data,))
        
        # Downstream gflops
        input_size = (num_enc_layers_dict[args.pretrain_model], 1, 50, hid_dim_dict[args.pretrain_model])
        input_data = torch.randn(input_size).cuda()
        downstream_flops, downstream_params = thop.profile(model, inputs=(input_data,))
        flops = (flops+downstream_flops) / 1e9  # Convert FLOPs to GFLOPs
    return flops

if __name__ == '__main__':

    # Argument parser
    args = parse_finetune_args()
    with open("../../config/config.yml", "r") as stream: config = yaml.safe_load(stream)
    args.split_dir      = str(Path(config["project_dir"]).joinpath("train_split"))
    args.data_dir       = str(Path(config["project_dir"]).joinpath("audio"))
    args.log_dir        = str(Path(config["project_dir"]).joinpath("finetune"))
    args.flops_dir      = str(Path(config["project_dir"]).joinpath("flops"))

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    # Number of folds in the exp
    total_folds = 2
    
    # Save results
    flops_res_dict = dict()
    for fold_idx in range(1, total_folds):
        
        # Read train/dev file list
        train_file_list, dev_file_list, test_file_list = load_finetune_audios(
            args.split_dir, audio_path=args.data_dir, dataset=args.dataset, fold_idx=fold_idx
        )
        
        return_dataset_stats(
            args.split_dir, dataset=args.dataset, fold_idx=fold_idx
        )
        
        # Log/model dir
        log_dir = Path(args.log_dir).joinpath(
            args.dataset, 
            args.pretrain_model,
            f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.downstream_model}_conv{args.conv_layers}_hid{args.hidden_size}_{args.pooling}'
        )

        # Fairness dir
        flops_dir = Path(args.flops_dir).joinpath(
            args.dataset, 
            args.pretrain_model,
            f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.downstream_model}_conv{args.conv_layers}_hid{args.hidden_size}_{args.pooling}'
        )
        Path.mkdir(flops_dir, parents=True, exist_ok=True)

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
            
        # Define the downstream models
        if args.downstream_model == "cnn":
            # Define the number of class
            if args.dataset in ["iemocap", "msp-improv", "meld", "iemocap_impro"]: num_class = 4
            elif args.dataset in ["msp-podcast"]: num_class = 4
            elif args.dataset in ["crema_d"]: num_class = 4
            elif args.dataset in ["ravdess"]: num_class = 7

            # Define the models
            model = CNNSelfAttention(
                input_dim=hid_dim_dict[args.pretrain_model], 
                output_class_num=num_class, 
                conv_layer=args.conv_layers, 
                num_enc_layers=num_enc_layers_dict[args.pretrain_model], 
                pooling_method=args.pooling
            )
            model.load_state_dict(
                torch.load(str(log_dir.joinpath(f'fold_{fold_idx}.pt'))), 
                strict=False
            )
            model = model.to(device)
        
        # Perform test
        gflops = validate_epoch(
            model, device, split="Test"
        )

        flops_res_dict[fold_idx] = dict()
        flops_res_dict[fold_idx]["flops"] = gflops
        
        # Save fairness results
        jsonString = json.dumps(flops_res_dict, indent=4)
        jsonFile = open(str(flops_dir.joinpath(f'gflops.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()
