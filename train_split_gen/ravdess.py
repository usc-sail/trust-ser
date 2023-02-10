import json
import numpy as np
import pandas as pd
import pickle, pdb, re, os

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
    data_path = Path('/media/data/public-data/SER/Ravdess')
    output_path = Path('/media/data/projects/speech-privacy/emo2vec')

    Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
    train_list, dev_list, test_list = list(), list(), list()
    
    # ravdess
    speaker_ids = [speaker_id for speaker_id in os.listdir(data_path) if '.zip' not in speaker_id]
    for speaker_id in speaker_ids:
        for file_name in os.listdir(data_path.joinpath(speaker_id)):
            actor_id = int(speaker_id.split('_')[1])
            label = int(file_name.split('-')[2])
            gender = 'female' if int(file_name.split('-')[-1].split('.')[0]) % 2 else 'male'
            file_path = data_path.joinpath(speaker_id, file_name)
            # [key, speaker id, gender, path, label]
            file_data = [file_name, f'ravdess_{speaker_id}', gender, str(file_path), label]
            
            # append data
            if actor_id > 19: test_list.append(file_data)
            elif actor_id <= 19 and actor_id > 14: dev_list.append(file_data)
            else: train_list.append(file_data)
    
    return_dict = dict()
    return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
    logging.info(f'-------------------------------------------------------')
    logging.info(f'Split distribution for RAVDESS dataset')
    for split in ['train', 'dev', 'test']:
        logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
    logging.info(f'-------------------------------------------------------')
    
    # dump the dictionary
    jsonString = json.dumps(return_dict, indent=4)
    jsonFile = open(str(output_path.joinpath('train_split', f'ravdess.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    