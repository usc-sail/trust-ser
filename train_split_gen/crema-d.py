import json
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
    data_path = Path('/media/data/public-data/SER/crema_d/CREMA-D/')
    output_path = Path('/media/data/projects/speech-privacy/emo2vec')

    Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
    train_list, dev_list, test_list = list(), list(), list()
    
    # crema-d
    file_list = [x for x in Path(data_path).joinpath("AudioWAV").iterdir() if '.wav' in x.parts[-1]]
    file_list.sort()

    demo_df = pd.read_csv(str(Path(data_path).joinpath('VideoDemographics.csv')), index_col=0)
    rating_df = pd.read_csv(str(Path(data_path).joinpath('processedResults', 'summaryTable.csv')), index_col=1)
    
    for idx, file_path in enumerate(file_list):
        if '1076_MTI_SAD_XX.wav' in str(file_path): continue
        sentence_file = file_path.parts[-1].split('.wav')[0]
        sentence_part = sentence_file.split('_')
        speaker_id = int(sentence_part[0])
        gender = 'M' if demo_df.loc[int(speaker_id), 'Sex'] == 'Male' else 'F'
        label = rating_df.loc[sentence_file, 'MultiModalVote']
        
        # [key, speaker id, gender, path, label]
        file_data = [sentence_file, f'crema_d_{speaker_id}', gender, str(file_path), label]

        # append data
        if speaker_id > 1070: test_list.append(file_data)
        elif speaker_id <= 1070 and speaker_id > 1050: dev_list.append(file_data)
        else: train_list.append(file_data)
    
    return_dict = dict()
    return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
    logging.info(f'-------------------------------------------------------')
    logging.info(f'Split distribution for CREMA-D dataset')
    for split in ['train', 'dev', 'test']:
        logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
    logging.info(f'-------------------------------------------------------')
    
    # dump the dictionary
    jsonString = json.dumps(return_dict, indent=4)
    jsonFile = open(str(output_path.joinpath('train_split', f'crema_d.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    