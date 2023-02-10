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
    data_path = Path('/media/data/sail-data/Recola')
    output_path = Path('/media/data/projects/speech-privacy/emo2vec')

    Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
    train_list, dev_list, test1_list, test2_list = list(), list(), list(), list()
    
    # recola
    sub_id_list = [sub_id.split('.')[0] for sub_id in os.list_dir(data_path.joinpath('RECOLA-Annotation'))]
    for sub_id in sub_id_list:
        gt_df = pd.read_csv(str(Path(data_path).joinpath('RECOLA-Annotation', f'{sub_id}.csv')), index_col=None)
        audio_df = pd.read_csv(str(Path(data_path).joinpath('RECOLA-Audio_timings', f'{sub_id}.csv')), index_col=None)
        
    for idx in range(len(gt_df)):
        file_name = gt_df.id.values[idx]
        start = gt_df.start.values[idx]
        end = gt_df.end.values[idx]
        sentiment = gt_df.sentiment.values[idx]
        
        # [key, speaker id, gender, path, label]
        file_path = data_path.joinpath('Audio', 'Full', 'WAV_16000', f'{file_name}.wav')
        file_data = [file_name, file_name, '', str(file_path), [start, end], sentiment]
        if file_name in standard_train_fold: train_list.append(file_data)
        elif file_name in standard_valid_fold: dev_list.append(file_data)
        elif file_name in standard_test_fold: test_list.append(file_data)
        
    return_dict = dict()
    return_dict['train'], return_dict['dev'], return_dict['test1'], return_dict['test2'] = train_list, dev_list, test1_list, test2_list
    logging.info(f'-------------------------------------------------------')
    logging.info(f'Split distribution for MSP-podcast dataset')
    for split in ['train', 'dev', 'test1', 'test2']:
        logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
    logging.info(f'-------------------------------------------------------')
    
    # dump the dictionary
    jsonString = json.dumps(return_dict, indent=4)
    jsonFile = open(str(output_path.joinpath('train_split', f'msp-podcast.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    