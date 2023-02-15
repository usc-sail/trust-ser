# Trustworthy Speech Emotion Recognition
#### Trust-SER is an open source project for researchers exploring SER applications with trustworthiness elements

The core elements for Trustworthy Speech Emotion Recognition:

<div align="center">
 <img src="img/trustworthy-ser.png" width="750px">
</div>

To begin with, please clone this repo:
```
git clone git@github.com:usc-sail/trust-ser.git
```

To install the conda environment:
```
cd trust-ser
conda env create -f trust-ser.yml
conda activate trust-ser
```

Please specify the data file to your work dir under config/config.yml

```
data_dir:
  crema_d: CREMA_D_PATH
  iemocap: IEMOCAP_PATH
  meld: MELD_PATH
  msp-improv: MSP-IMPROV_PATH
  msp-podcast: MSP-PODCAST_PATH
  ravdess: RAVDESS_PATH
project_dir: OUTPUT_PATH
```

### Data Spliting
For most of the dataset, user need first split the train/dev/test by the given script file. Take the IEMOCAP data as instance:
```
cd train_split_gen
python3 iemocap.py
```

### Audio Preprocess
For most of the dataset, user can generate the preprocessed audio file by the given script file. The preprocessing includes resample to 16kHz and to mono channel. Take the IEMOCAP data as instance:
```
cd preprocess_audio
python3 preprocess_audio.py --dataset iemocap
# dataset: iemocap, ravdess, msp-improv, msp-podcast, crema_d
```

### ML training
To train with a pretrained backbone, use the following:
```
cd experiment
CUDA_VISIBLE_DEVICES=0 taskset -c 1-30 python3 finetune_single_thread.py --pretrain_model apc --dataset iemocap --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128

# pretrain_model: apc, tera, wavlm, wav2vec2_0, whisper (to add)
# pooling: mean, att (self-attention)
# hidden_size: size of cnn
# conv_layers: number of cnn layers
```
