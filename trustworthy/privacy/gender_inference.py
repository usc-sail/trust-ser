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
from dataloader import load_finetune_audios, set_gender_dataloader, return_weights
from pretrained_backbones import Wav2Vec, APC, TERA, WavLM, WhisperTiny, WhisperBase, WhisperSmall

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)

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

os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1" 
os.environ["OMP_NUM_THREADS"] = "1" 

def train_epoch(
    dataloader, 
    model, 
    device, 
    optimizer
):
    model.train()
    backbone_model.train()
    criterion = nn.NLLLoss().to(device)
    eval_metric = EvalMetric()
    
    logging.info(f'-------------------------------------------------------------------')
    for batch_idx, batch_data in enumerate(dataloader):
        # Read data
        model.zero_grad()
        optimizer.zero_grad()
        x, y, length = batch_data
        x, y, length = x.to(device), y.to(device), length.to(device)
        
        # Forward pass
        with torch.no_grad():
            feat, length, mask = backbone_model(x, norm=args.norm, length=length)
        outputs = model(feat, length, mask)
        outputs = torch.log_softmax(outputs, dim=1)
                    
        # Calculate Loss
        loss = criterion(outputs, y)

        # Backward
        loss.backward()
        
        # Clip gradients
        optimizer.step()
        
        eval_metric.append_classification_results(y, outputs, loss)
        
        if (batch_idx % 10 == 0 and batch_idx != 0) or batch_idx == len(dataloader) - 1:
            result_dict = eval_metric.classification_summary()
            logging.info(f'Fold {fold_idx} - Current Train Loss at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["loss"]:.3f}')
            logging.info(f'Fold {fold_idx} - Current Train UAR at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["uar"]:.2f}%')
            logging.info(f'Fold {fold_idx} - Current Train ACC at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {result_dict["acc"]:.2f}%')
            logging.info(f'Fold {fold_idx} - Current Train LR at epoch {epoch}, step {batch_idx+1}/{len(dataloader)}: {scheduler.optimizer.param_groups[0]["lr"]}')
            logging.info(f'-------------------------------------------------------------------')
    
    logging.info(f'-------------------------------------------------------------------')
    result_dict = eval_metric.classification_summary()
    return result_dict
    
def validate_epoch(
    dataloader, 
    model, 
    device,
    split:  str="Validation"
):
    model.eval()
    backbone_model.eval()
    criterion = nn.NLLLoss().to(device)
    eval_metric = EvalMetric()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # read data
            x, y = batch_data
            x, y = x.to(device), y.to(device)
            
            # forward pass
            feat = backbone_model(x, norm=args.norm)
            outputs = model(feat)
            outputs = torch.log_softmax(outputs, dim=1)
                        
            # backward
            loss = criterion(outputs, y)
            eval_metric.append_classification_results(y, outputs, loss)
        
            if (batch_idx % 50 == 0 and batch_idx != 0) or batch_idx == len(dataloader) - 1:
                result_dict = eval_metric.classification_summary()
                logging.info(f'Fold {fold_idx} - Current {split} Loss at epoch {epoch}, step {batch_idx+1}/{len(dataloader)} {result_dict["loss"]:.3f}')
                logging.info(f'Fold {fold_idx} - Current {split} UAR at epoch {epoch}, step {batch_idx+1}/{len(dataloader)} {result_dict["uar"]:.2f}%')
                logging.info(f'Fold {fold_idx} - Current {split} ACC at epoch {epoch}, step {batch_idx+1}/{len(dataloader)} {result_dict["acc"]:.2f}%')
                logging.info(f'-------------------------------------------------------------------')
    logging.info(f'-------------------------------------------------------------------')
    result_dict = eval_metric.classification_summary()
    if split == "Validation": scheduler.step(result_dict["loss"])
    return result_dict


if __name__ == '__main__':

    # Argument parser
    args = parse_finetune_args()
    with open("../../config/config.yml", "r") as stream: config = yaml.safe_load(stream)
    args.split_dir      = str(Path(config["project_dir"]).joinpath("train_split"))
    args.data_dir       = str(Path(config["project_dir"]).joinpath("audio"))
    args.log_dir        = str(Path(config["project_dir"]).joinpath("finetune"))
    args.attack_dir     = str(Path(config["project_dir"]).joinpath("privacy"))

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    best_dict = dict()
    # We perform 1 fold for gender inference
    for fold_idx in range(1, 2):

        # Read train/dev file list
        train_file_list, dev_file_list, test_file_list = load_finetune_audios(
            args.split_dir, audio_path=args.data_dir, dataset=args.dataset, fold_idx=fold_idx
        )
    
        # Set train/dev/test dataloader
        train_dataloader = set_gender_dataloader(
            args, train_file_list, is_train=True
        )
        dev_dataloader = set_gender_dataloader(
            args, dev_file_list, is_train=False
        )
        test_dataloader = set_gender_dataloader(
            args, test_file_list, is_train=False
        )

        # Log/model dir
        log_dir = Path(args.log_dir).joinpath(
            args.dataset, 
            args.pretrain_model,
            f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.downstream_model}_conv{args.conv_layers}_hid{args.hidden_size}_{args.pooling}_{args.finetune}'
        )
        
        # Define attack dir
        attack_dir = Path(args.attack_dir).joinpath(
            args.privacy_attack, args.dataset, args.pretrain_model,
            f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.downstream_model}_conv{args.conv_layers}_hid{args.hidden_size}_{args.pooling}_{args.finetune}'
        )
        Path.mkdir(attack_dir, parents=True, exist_ok=True)
        
        # Set seeds
        set_seed(8*fold_idx)
        
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
            # Define the models
            model = CNNSelfAttention(
                input_dim=hid_dim_dict[args.pretrain_model], 
                output_class_num=2, 
                conv_layer=args.conv_layers, 
                num_enc_layers=num_enc_layers_dict[args.pretrain_model], 
                pooling_method=args.pooling
            ).to(device)
            
        if args.finetune != "frozen":
            backbone_model.load_state_dict(
                torch.load(str(log_dir.joinpath(f'fold_{fold_idx}_backbone.pt'))), 
                strict=False
            )
        
        # Read trainable params
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f'Trainable params size: {params/(1024*1024):.2f} M')
        
        # Define optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=0.0005, 
            weight_decay=1e-4,
            betas=(0.9, 0.98)
        )

        # Define scheduler, patient = 5, minimum learning rate 5e-5
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=5e-5
        )

        # Training steps
        best_dev_uar, best_test_uar, best_epoch = 0, 0, 0
        best_dev_acc, best_test_acc = 0, 0
        
        result_hist_dict = dict()
        for epoch in range(10):
            train_result = train_epoch(
                train_dataloader, model, device, optimizer
            )

            dev_result = validate_epoch(
                dev_dataloader, model, device
            )
            
            test_result = validate_epoch(
                test_dataloader, model, device, split="Test"
            )
            # if we get a better results
            if best_dev_uar < dev_result["uar"]:
                best_dev_uar = dev_result["uar"]
                best_test_uar = test_result["uar"]
                best_dev_acc = dev_result["acc"]
                best_test_acc = test_result["acc"]
                best_epoch = epoch
                torch.save(model.state_dict(), str(attack_dir.joinpath(f'fold_{fold_idx}.pt')))
            
            logging.info(f'-------------------------------------------------------------------')
            logging.info(f"Fold {fold_idx} - Best train epoch {best_epoch}, best dev UAR {best_dev_uar:.2f}%, best test UAR {best_test_uar:.2f}%")
            logging.info(f"Fold {fold_idx} - Best train epoch {best_epoch}, best dev ACC {best_dev_acc:.2f}%, best test ACC {best_test_acc:.2f}%")
            logging.info(f'-------------------------------------------------------------------')
            
            # log the current result
            log_epoch_result(result_hist_dict, epoch, train_result, dev_result, test_result, attack_dir, fold_idx)

        # log the best results
        log_best_result(result_hist_dict, epoch, best_dev_uar, best_dev_acc, best_test_uar, best_test_acc, attack_dir, fold_idx)
        
        best_dict[fold_idx] = dict()
        best_dict[fold_idx]["uar"] = best_test_uar
        best_dict[fold_idx]["acc"] = best_test_acc
        
        # save best results
        jsonString = json.dumps(best_dict, indent=4)
        jsonFile = open(str(attack_dir.joinpath(f'results.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    uar_list = [best_dict[fold_idx]["uar"] for fold_idx in best_dict]
    acc_list = [best_dict[fold_idx]["acc"] for fold_idx in best_dict]
    best_dict["average"] = dict()
    best_dict["average"]["uar"] = np.mean(uar_list)
    best_dict["average"]["acc"] = np.mean(acc_list)
    
    best_dict["std"] = dict()
    best_dict["std"]["uar"] = np.std(uar_list)
    best_dict["std"]["acc"] = np.std(acc_list)
    
    # save best results
    jsonString = json.dumps(best_dict, indent=4)
    jsonFile = open(str(attack_dir.joinpath(f'results.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()
