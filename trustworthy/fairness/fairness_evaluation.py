import json
import yaml
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

def validate_epoch(
    dataloader, 
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
        for batch_idx, batch_data in enumerate(dataloader):
            # Read data
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            feat = backbone_model(x, norm=args.norm)
            outputs = model(feat)
            outputs = torch.log_softmax(outputs, dim=1)

            # Read gender and speaker id
            speaker_id = test_file_list[batch_idx][1]
            gender = test_file_list[batch_idx][2]
                        
            # backward
            eval_metric.append_classification_results(
                y, outputs, loss=None, demographics=gender, speaker_id=speaker_id
            )
            
            if (batch_idx % 50 == 0 and batch_idx != 0) or batch_idx == len(dataloader) - 1:
                result_dict = eval_metric.classification_summary()
                logging.info(f'-------------------------------------------------------------------')
                logging.info(f'Fold {fold_idx} - Current UAR, step {batch_idx+1}/{len(dataloader)} {result_dict["uar"]:.2f}%')
                logging.info(f'Fold {fold_idx} - Current ACC, step {batch_idx+1}/{len(dataloader)} {result_dict["acc"]:.2f}%')
                logging.info(f'-------------------------------------------------------------------')
    logging.info(f'-------------------------------------------------------------------')
    demographic_parity = eval_metric.demographic_parity()
    statistical_parity = eval_metric.statistical_parity()
    equality_of_opp = eval_metric.equality_of_opp()
    return demographic_parity, statistical_parity, equality_of_opp


if __name__ == '__main__':

    # Argument parser
    args = parse_finetune_args()
    with open("../../config/config.yml", "r") as stream: config = yaml.safe_load(stream)
    args.split_dir      = str(Path(config["project_dir"]).joinpath("train_split"))
    args.data_dir       = str(Path(config["project_dir"]).joinpath("audio"))
    args.log_dir        = str(Path(config["project_dir"]).joinpath("finetune"))
    args.fairness_dir   = str(Path(config["project_dir"]).joinpath("fairness"))

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    # Number of folds in the exp
    if args.dataset == "msp-improv": total_folds = 7
    else: total_folds = 6
    
    # Save results
    fairness_res_dict = dict()
    for fold_idx in range(1, total_folds):
        
        # Read train/dev file list
        train_file_list, dev_file_list, test_file_list = load_finetune_audios(
            args.split_dir, audio_path=args.data_dir, dataset=args.dataset, fold_idx=fold_idx
        )
        
        # Set test dataloader
        test_dataloader = set_finetune_dataloader(
            args, test_file_list, is_train=False
        )

        # Log/model dir
        log_dir = Path(args.log_dir).joinpath(
            args.dataset, 
            args.pretrain_model,
            f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.downstream_model}_conv{args.conv_layers}_hid{args.hidden_size}_{args.pooling}_{args.finetune}'
        )

        # Fairness dir
        fairness_dir = Path(args.fairness_dir).joinpath(
            args.dataset, 
            args.pretrain_model,
            f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.downstream_model}_conv{args.conv_layers}_hid{args.hidden_size}_{args.pooling}_{args.finetune}'
        )
        Path.mkdir(fairness_dir, parents=True, exist_ok=True)

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
            
            if args.finetune != "frozen":
                backbone_model.load_state_dict(
                    torch.load(str(log_dir.joinpath(f'fold_{fold_idx}_backbone.pt'))), 
                    strict=False
                )
        
        # Perform test
        demographic_parity, statistical_parity, equality_of_opp = validate_epoch(
            test_dataloader, model, device, split="Test"
        )

        fairness_res_dict[fold_idx] = dict()
        fairness_res_dict[fold_idx]["demographic_parity"] = demographic_parity
        fairness_res_dict[fold_idx]["statistical_parity"] = statistical_parity
        fairness_res_dict[fold_idx]["equality_of_opp"] = equality_of_opp

        # Save fairness results
        jsonString = json.dumps(fairness_res_dict, indent=4)
        jsonFile = open(str(fairness_dir.joinpath(f'results.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    # Save average results
    demographic_parity_list = [fairness_res_dict[fold_idx]["demographic_parity"] for fold_idx in fairness_res_dict]
    statistical_parity_list = [fairness_res_dict[fold_idx]["statistical_parity"] for fold_idx in fairness_res_dict]
    equality_of_opp_list = [fairness_res_dict[fold_idx]["equality_of_opp"] for fold_idx in fairness_res_dict]
    fairness_res_dict["average"], fairness_res_dict["std"] = dict(), dict()
    fairness_res_dict["average"]["demographic_parity"] = np.mean(demographic_parity_list)
    fairness_res_dict["average"]["statistical_parity"] = np.mean(statistical_parity_list)
    fairness_res_dict["average"]["equality_of_opp"] = np.mean(equality_of_opp_list)
    fairness_res_dict["std"]["demographic_parity"] = np.std(demographic_parity_list)
    fairness_res_dict["std"]["statistical_parity"] = np.std(statistical_parity_list)
    fairness_res_dict["std"]["equality_of_opp"] = np.std(equality_of_opp_list)
    
    # Save fairness results
    jsonString = json.dumps(fairness_res_dict, indent=4)
    jsonFile = open(str(fairness_dir.joinpath(f'results.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()
