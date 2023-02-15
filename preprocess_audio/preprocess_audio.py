import json
import torch
import torchaudio
import numpy as np
import pandas as pd
import pickle, pdb, re

from tqdm import tqdm
from pathlib import Path

# define logging console
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-3s ==> %(message)s', 
    level=logging.INFO, 
    datefmt='%Y-%m-%d %H:%M:%S'
)


if __name__ == '__main__':

    # data path
    with open("../config/config.yml", "r") as stream:
         config = yaml.safe_load(stream)
    data_path   = Path(config["data_dir"]["crema_d"])
    split_path = Path(config["project_dir"]).joinpath("train_split")
    audio_path = Path(config["project_dir"]).joinpath("audio")

    # parser
    parser = argparse.ArgumentParser(description='input dataset')
    parser.add_argument(
        '--dataset', 
        default='iemocap',
        type=str, 
        help='dataset name'
    )
    args = parser.parse_args()

    # args.dataset
    # read splits
    # for dataset in ['iemocap']:
    # for dataset in ['msp-improv']:
    # for dataset in ['meld']:
    # for dataset in ['crema_d']:
    # for dataset in ['ravdess']:
    # for dataset in ['emov_db']:
    # for dataset in ['vox-movie']:
    # for dataset in ['msp-podcast']:
    # for dataset in ['cmu-mosei']:
    if args.dataset in ["iemocap", "crema_d", "ravdess", "msp-improv"]:
        with open(str(split_path.joinpath(f'{args.dataset}_fold1.json')), "r") as f: split_dict = json.load(f)
    else:
        with open(str(split_path.joinpath(f'{args.dataset}.json')), "r") as f: split_dict = json.load(f)
    for split in ['train', 'dev', 'test']:
        Path.mkdir(audio_path.joinpath(args.dataset), parents=True, exist_ok=True)
        # for idx, data in tqdm(enumerate(), ncols=10, miniters=10):
        for idx in tqdm(range(len(split_dict[split]))):
            data = split_dict[split][idx]
            file_path = data[3]
            speaker_id = data[1]
            # if dataset in ['iemocap', 'msp-improv', 'meld', 'crema_d', 'ravdess', 'emov_db']:
            waveform, sample_rate = torchaudio.load(str(file_path))
            
            if waveform.shape[0] != 1:
                waveform = torch.mean(waveform, dim=0).unsqueeze(0)
            if sample_rate != 16000:
                transform_model = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = transform_model(waveform)
            if args.dataset == 'cmu-mosei':
                start, end = int(data[4][0] * 16000), int(data[4][1] * 16000)
                waveform = waveform[:, start:end]

            if args.dataset in ['iemocap', 'msp-improv', 'meld', 'crema_d', 'msp-podcast']:
                output_path = audio_path.joinpath(args.dataset, file_path.split('/')[-1])
            elif args.dataset in ['ravdess', 'emov_db', 'vox-movie']:
                output_path = audio_path.joinpath(args.dataset, f'{speaker_id}_{file_path.split("/")[-1]}')
            elif args.dataset in ['cmu-mosei']:
                output_path = audio_path.joinpath(args.dataset, f'{str(start).replace(".", "_")}_{file_path.split("/")[-1]}')
        
            torchaudio.save(str(output_path), waveform, 16000)
            split_dict[split][idx][3] = str(output_path)
                
        logging.info(f'-------------------------------------------------------')
        logging.info(f'Preprocess audio for {dataset} dataset')
        for split in ['train', 'dev', 'test']: logging.info(f'Split {split}: Number of files {len(split_dict[split])}')
        logging.info(f'-------------------------------------------------------')
