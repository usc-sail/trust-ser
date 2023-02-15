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
from pretrained_backbones import Wav2Vec, APC, TERA, WavLM
from downstream_models import DNNClassifier, CNNSelfAttention
from dataloader import load_finetune_audios, set_finetune_dataloader, return_weights

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
    "wavlm":        768,
    "apc":          512,
}

num_enc_layers_dict = {
    "wav2vec2_0":   12,
    "wavlm":        12,
    "tera":         4,
    "apc":          3,
}

def train_epoch(
    dataloader, 
    model, 
    device, 
    optimizer
):
    model.train()
    backbone_model.train()
    criterion = nn.NLLLoss(weights).to(device)
    eval_metric = EvalMetric()
    
    logging.info(f'-------------------------------------------------------------------')
    for batch_idx, batch_data in enumerate(dataloader):
        # read data
        model.zero_grad()
        optimizer.zero_grad()
        x, y, length = batch_data
        x, y, length = x.to(device), y.to(device), length.to(device)
        
        # forward pass
        feat, length, mask = backbone_model(x, norm=args.norm, length=length)
        outputs = model(feat, length, mask)
        outputs = torch.log_softmax(outputs, dim=1)
                    
        # backward
        loss = criterion(outputs, y)

        # backward
        loss.backward()
        
        # clip gradients
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
    criterion = nn.NLLLoss(weights).to(device)
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

    # argument parser
    args = parse_finetune_args()

    # find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    best_dict = dict()
    if args.dataset == "msp-improv": total_folds = 7
    else: total_folds = 6
    for fold_idx in range(1, total_folds):

        # read train/dev file list
        train_file_list, dev_file_list, test_file_list = load_finetune_audios(
            args.split_dir, dataset=args.dataset, fold_idx=fold_idx
        )
        weights = return_weights(
            args.split_dir, dataset=args.dataset, fold_idx=fold_idx
        )
    
        # train/dev/test dataloader
        train_dataloader = set_finetune_dataloader(
            args, train_file_list, is_train=True
        )
        dev_dataloader = set_finetune_dataloader(
            args, dev_file_list, is_train=False
        )
        test_dataloader = set_finetune_dataloader(
            args, test_file_list, is_train=False
        )

        # log dir
        log_dir = Path(args.log_dir).joinpath(
            args.dataset, 
            args.pretrain_model,
            f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.downstream_model}_conv{args.conv_layers}_hid{args.hidden_size}_{args.pooling}'
        )
        Path.mkdir(log_dir, parents=True, exist_ok=True)
        
        # set seeds
        set_seed(8*fold_idx)
        # Define the model
        if args.pretrain_model == "wav2vec2_0":
            # original wav2vec model
            backbone_model = Wav2Vec().to(device)
        elif args.pretrain_model == "apc":
            # APC wrapper from superb
            backbone_model = APC().to(device)
        elif args.pretrain_model == "tera":
            # TERA wrapper from superb
            backbone_model = TERA().to(device)
        elif args.pretrain_model == "wavlm":
            # wavlm wrapper from huggingface
            backbone_model = WavLM().to(device)
            
        # define the downstream models
        if args.downstream_model == "cnn":
            # define the number of class
            if args.dataset in ["iemocap", "msp-improv", "meld", "iemocap_impro"]: num_class = 4
            elif args.dataset in ["msp-podcast"]: num_class = 4
            elif args.dataset in ["crema_d"]: num_class = 4
            elif args.dataset in ["ravdess"]: num_class = 7

            # define the models
            model = CNNSelfAttention(
                input_dim=hid_dim_dict[args.pretrain_model], 
                output_class_num=num_class, 
                conv_layer=args.conv_layers, 
                num_enc_layers=num_enc_layers_dict[args.pretrain_model], 
                pooling_method=args.pooling
            ).to(device)
        
        # trainable params
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logging.info(f'Trainable params size: {params/(1024*1024):.2f} M')
        
        # optimizer
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.learning_rate, 
            weight_decay=1e-4,
            betas=(0.9, 0.98)
        )

        # scheduler, patient = 5, minimum learning rate 5e-5
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', patience=5, factor=0.5, verbose=True, min_lr=5e-5
        )

        # Training steps
        best_dev_uar, best_test_uar, best_epoch = 0, 0, 0
        best_dev_acc, best_test_acc = 0, 0
        
        result_hist_dict = dict()
        for epoch in range(args.num_epochs):
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
                torch.save(model.state_dict(), str(log_dir.joinpath(f'fold_{fold_idx}.pt')))
            
            logging.info(f'-------------------------------------------------------------------')
            logging.info(f"Fold {fold_idx} - Best train epoch {best_epoch}, best dev UAR {best_dev_uar:.2f}%, best test UAR {best_test_uar:.2f}%")
            logging.info(f"Fold {fold_idx} - Best train epoch {best_epoch}, best dev ACC {best_dev_acc:.2f}%, best test ACC {best_test_acc:.2f}%")
            logging.info(f'-------------------------------------------------------------------')
            
            # log the current result
            log_epoch_result(result_hist_dict, epoch, train_result, dev_result, test_result, log_dir, fold_idx)

        # log the best results
        log_best_result(result_hist_dict, epoch, best_dev_uar, best_dev_acc, best_test_uar, best_test_acc, log_dir, fold_idx)
        
        best_dict[fold_idx] = dict()
        best_dict[fold_idx]["uar"] = best_test_uar
        best_dict[fold_idx]["acc"] = best_test_acc
        
        # save best results
        jsonString = json.dumps(best_dict, indent=4)
        jsonFile = open(str(log_dir.joinpath(f'results.json')), "w")
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
    jsonFile = open(str(log_dir.joinpath(f'results.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()
