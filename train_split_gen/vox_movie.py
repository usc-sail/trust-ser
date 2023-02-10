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
    data_path = Path('/media/data/public-data/SER/VoxMovies/vox_movies')
    output_path = Path('/media/data/projects/speech-privacy/emo2vec')

    Path.mkdir(output_path.joinpath('train_split'), parents=True, exist_ok=True)
    train_list, dev_list, test_list = list(), list(), list()
    
    # vox-movie
    for speaker_id in os.listdir(data_path.joinpath('voxmovies_train')):
        if '.' in speaker_id: continue
        for movie_id in os.listdir(data_path.joinpath('voxmovies_train', speaker_id)):
            if movie_id[0] == '.': continue
            for file_name in os.listdir(data_path.joinpath('voxmovies_train', speaker_id, movie_id)):
                if file_name[0] == '.': continue
                # [key, speaker id, gender, path, label]
                if '.wav' not in file_name: continue
                file_path = data_path.joinpath('voxmovies_train', speaker_id, movie_id, file_name)
                file_data = [file_name, f'voxmovies_{speaker_id}', '', str(file_path), '']
                train_list.append(file_data)
    
    for speaker_id in os.listdir(data_path.joinpath('voxmovies_test')):
        if '.' in speaker_id: continue
        for movie_id in os.listdir(data_path.joinpath('voxmovies_test', speaker_id)):
            if movie_id[0] == '.': continue
            for file_name in os.listdir(data_path.joinpath('voxmovies_test', speaker_id, movie_id)):
                if file_name[0] == '.': continue
                # [key, speaker id, gender, path, label]
                file_path = data_path.joinpath('voxmovies_test', speaker_id, movie_id, file_name)
                file_data = [file_name, f'voxmovies_{speaker_id}', '', str(file_path), '']
                train_list.append(file_data)
    
    return_dict = dict()
    return_dict['train'], return_dict['dev'], return_dict['test'] = train_list, dev_list, test_list
    logging.info(f'-------------------------------------------------------')
    logging.info(f'Split distribution for Vox-Movie dataset')
    for split in ['train', 'dev', 'test']:
        logging.info(f'Split {split}: Number of files {len(return_dict[split])}')
    logging.info(f'-------------------------------------------------------')
    
    # dump the dictionary
    jsonString = json.dumps(return_dict, indent=4)
    jsonFile = open(str(output_path.joinpath('train_split', f'vox-movie.json')), "w")
    jsonFile.write(jsonString)
    jsonFile.close()

    