import json
import yaml
import torch
import random
import torchaudio
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
from dataloader import load_finetune_audios, set_finetune_dataloader, return_weights
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

# PGD
def pgd_attack(sound, ori_sound, eps, alpha, data_grad):
    adv_sound = sound - alpha * data_grad.sign() # + -> - !!!
    eta = torch.clamp(adv_sound - ori_sound.data, min=-eps, max=eps)
    sound = ori_sound + eta
    return sound

# FGSM
def fgsm_attack(sound, data_grad):
    # Find direction of gradient
    sign_data_grad = data_grad.sign()
    
    # Calculate the gradients
    sound_norm = np.sqrt(np.mean(np.square(sound.detach().cpu().numpy())))
    noise_rms = np.sqrt(np.mean(np.square(sign_data_grad.detach().cpu().numpy())))
    desired_noise_rms = calculate_desired_noise_rms(sound_norm, snr=args.snr)

    # Adjust the noise to match the desired noise RMS
    noise_sound = sign_data_grad * (desired_noise_rms / noise_rms)
    
    # add noise "epilon * direction" to the ori sound
    perturbed_sound = sound + noise_sound
    return perturbed_sound

# The code was taken: 
# https://github.com/iver56/audiomentations/blob/4e3685491aabb5b2e24191020f7b0a8356d5feec/audiomentations/core/utils.py#L112
def calculate_desired_noise_rms(clean_rms, snr):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.
    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20
    :param clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
    :param snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60
    :return:
    """
    a = float(snr) / 20
    noise_rms = clean_rms / (10**a)
    return noise_rms

def validate_epoch(
    dataloader, 
    model, 
    device,
    split:  str="Validation"
):
    eval_metric = EvalMetric()
    criterion = nn.NLLLoss().to(device)
    
    total_attacks, success_attack = 0, 0
    for batch_idx, batch_data in enumerate(dataloader):
        # Read data
        x, y = batch_data
        x, y = x.to(device), y.to(device)

        # 1. Only perform attack on the correct samples
        # Forward pass
        model.eval()
        backbone_model.eval()
        feat = backbone_model(x, norm=args.norm, is_attack=False)
        # feat = backbone_model(perturbed_x, norm=args.norm, is_attack=False)
        outputs = model(feat)
        outputs = torch.log_softmax(outputs, dim=1)
        if (y[0]==outputs.argmax()).detach().cpu().numpy() == False: 
            continue
        
        # FGSM attack
        total_attacks += 1
        if args.attack_method == "fgsm":
            # Perform Attack
            model.train()
            backbone_model.train()
        
            # Set require gradients to be true
            x.requires_grad = True
            feat = backbone_model(x, is_attack=True)
            outputs = model(feat)
            outputs = torch.log_softmax(outputs, dim=1)         
            # Backward
            loss = criterion(outputs, y)
            loss.backward()
            # Read gradients of data and perturb
            x_grad = x.grad.data
            perturbed_x = fgsm_attack(x, data_grad=x_grad)
            
            # torchaudio.save('original.wav', x.detach().cpu(), 16000)
            # torchaudio.save('adv.wav', perturbed_x.detach().cpu(), 16000)
            # pdb.set_trace()

            # Zero the gradients
            backbone_model.zero_grad()
            model.zero_grad()
        # PGD attack
        elif args.attack_method == "pgd":
            # Perform Attack
            model.train()
            backbone_model.train()
        
            for step in range(10):
                # Forward pass
                feat = backbone_model(x, norm=args.norm, is_attack=True)
                outputs = model(feat)
                outputs = torch.log_softmax(outputs, dim=1)         
                # backward
                loss = criterion(outputs, y)
                loss.backward()

        # Perform evaluation
        model.eval()
        backbone_model.eval()
        # Forward pass
        if args.attack_method in ["fgsm", "pgd"]: 
            feat = backbone_model(perturbed_x, norm=args.norm, is_attack=False)
        elif args.attack_method in ["guassian_noise"]:
            feat = backbone_model(x, norm=args.norm, is_attack=False)
        # feat = backbone_model(perturbed_x, norm=args.norm, is_attack=False)
        outputs = model(feat)
        outputs = torch.log_softmax(outputs, dim=1)

        if (y[0]==outputs.argmax()).detach().cpu().numpy() == False: 
            success_attack += 1

        if (batch_idx % 50 == 0 and batch_idx != 0) or batch_idx == len(dataloader) - 1:
            logging.info(f'-------------------------------------------------------------------')
            logging.info(f'Fold {fold_idx} - Current {split} Attack Success Rate step {batch_idx+1}/{len(dataloader)} {(success_attack / total_attacks)*100:.2f}%')
            logging.info(f'-------------------------------------------------------------------')
    logging.info(f'-------------------------------------------------------------------')
    attack_success_rate = (success_attack / total_attacks) * 100
    return attack_success_rate


if __name__ == '__main__':

    # Argument parser
    args = parse_finetune_args()
    with open("../../config/config.yml", "r") as stream: config = yaml.safe_load(stream)
    args.split_dir      = str(Path(config["project_dir"]).joinpath("train_split"))
    args.data_dir       = str(Path(config["project_dir"]).joinpath("audio"))
    args.log_dir        = str(Path(config["project_dir"]).joinpath("finetune"))
    args.attack_dir     = str(Path(config["project_dir"]).joinpath("attack"))

    # Find device
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')
    
    result_dict = dict()
    if args.dataset == "msp-improv": total_folds = 7
    else: total_folds = 6

    # Set flag for attack or not
    if args.attack_method in ["fgsm", "pgd"]: 
        is_attack               = True
        apply_guassian_noise    = False
    elif args.attack_method in ["guassian_noise"]:
        is_attack               = False
        apply_guassian_noise    = True

    # We perform 5 folds (6 folds only on msp-improv data with 6 sessions)
    for fold_idx in range(1, total_folds):

        # Read train/dev file list
        train_file_list, dev_file_list, test_file_list = load_finetune_audios(
            args.split_dir, audio_path=args.data_dir, dataset=args.dataset, fold_idx=fold_idx
        )
        
        test_dataloader = set_finetune_dataloader(
            args, test_file_list, is_train=False, apply_guassian_noise=apply_guassian_noise
        )

        # Define log dir
        log_dir = Path(args.log_dir).joinpath(
            args.dataset, args.pretrain_model,
            f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.downstream_model}_conv{args.conv_layers}_hid{args.hidden_size}_{args.pooling}_{args.finetune}'
        )
        # Define attack dir
        attack_dir = Path(args.attack_dir).joinpath(
            args.attack_method, str(args.snr), args.dataset, args.pretrain_model,
            f'lr{str(args.learning_rate).replace(".", "")}_ep{args.num_epochs}_{args.downstream_model}_conv{args.conv_layers}_hid{args.hidden_size}_{args.pooling}_{args.finetune}'
        )
        Path.mkdir(attack_dir, parents=True, exist_ok=True)
        
        # Set seeds
        set_seed(8*fold_idx)
        
        # Define the model wrapper
        if args.pretrain_model == "wav2vec2_0":
            # Wav2vec2_0 Wrapper
            backbone_model = Wav2Vec(is_attack=True).to(device)
        elif args.pretrain_model == "apc":
            # APC wrapper from superb
            backbone_model = APC().to(device)
        elif args.pretrain_model == "tera":
            # TERA wrapper from superb
            backbone_model = TERA().to(device)
        elif args.pretrain_model == "wavlm":
            # Wavlm wrapper from huggingface
            backbone_model = WavLM(is_attack=True).to(device)
        elif args.pretrain_model == "whisper_tiny":
            # Whisper tiny wrapper from huggingface
            backbone_model = WhisperTiny(is_attack=True).to(device)
        elif args.pretrain_model == "whisper_base":
            # Whisper base wrapper from huggingface
            backbone_model = WhisperBase(is_attack=True).to(device)
        elif args.pretrain_model == "whisper_small":
            # Whisper small wrapper from huggingface
            backbone_model = WhisperSmall(is_attack=True).to(device)

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
        
        attack_success_rate = validate_epoch(
            test_dataloader, model, device, split="Test"
        )
        
        result_dict[fold_idx] = dict()
        result_dict[fold_idx]["attack_success_rate"] = attack_success_rate
        
        # save best results
        jsonString = json.dumps(result_dict, indent=4)
        jsonFile = open(str(attack_dir.joinpath(f'results.json')), "w")
        jsonFile.write(jsonString)
        jsonFile.close()

    attack_success_list = [result_dict[fold_idx]["attack_success_rate"] for fold_idx in result_dict]
    result_dict["average"], result_dict["std"] = dict(), dict()
    result_dict["average"]["attack_success_rate"] = np.mean(attack_success_list)
    result_dict["std"]["attack_success_rate"] = np.std(attack_success_list)
    
    # save best results
    jsonString = json.dumps(result_dict, indent=4)
    jsonFile = open(str(attack_dir.joinpath(f'results.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()
