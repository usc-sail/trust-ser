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
    data_path = Path('/media/data/sail-data/iemocap/')
    output_path = Path('/media/data/projects/speech-privacy/emo2vec')

    Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
    train_list, dev_list, test_list = list(), list(), list()
    
    # iemocap 
    for session_id in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
        ground_truth_path_list = list(Path(data_path).joinpath(session_id, 'dialog', 'EmoEvaluation').glob('*.txt'))
        for ground_truth_path in tqdm(ground_truth_path_list, ncols=100, miniters=100):
            with open(str(ground_truth_path)) as f:
                file_content = f.read()
                useful_regex = re.compile(r'\[.+\]\n', re.IGNORECASE)
                label_lines = re.findall(useful_regex, file_content)
                for line in label_lines:
                    if 'Ses' in line:
                        sentence_file = line.split('\t')[-3]
                        gender = sentence_file.split('_')[-1][0]
                        speaker_id = 'iemocap_' + sentence_file.split('_')[0][:-1] + gender
                        label = line.split('\t')[-2]

                        file_path = Path(data_path).joinpath(
                            session_id, 'sentences', 'wav', '_'.join(sentence_file.split('_')[:-1]), f'{sentence_file}.wav'
                        )
                        # [key, speaker id, gender, path, label]
                        file_data = [sentence_file, speaker_id, gender, str(file_path), label]

                        # append data
                        if session_id == 'Session5': test_list.append(file_data)
                        elif session_id == 'Session4': dev_list.append(file_data)
                        else: train_list.append(file_data)

    return_dict = dict()
    return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
    logging.info(f'-------------------------------------------------------')
    logging.info(f'Split distribution for IEMOCAP dataset')
    for split in ['train', 'dev', 'test']:
        logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
    logging.info(f'-------------------------------------------------------')
    
    # dump the dictionary
    jsonString = json.dumps(return_dict, indent=4)
    jsonFile = open(str(output_path.joinpath('train_split', f'iemocap.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    