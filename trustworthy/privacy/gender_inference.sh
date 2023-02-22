# for pretrain_model in apc whisper_base whisper_tiny whisper_small wav2vec2_0 tera wavlm; do
# for pretrain_model in apc whisper_base whisper_tiny whisper_small wav2vec2_0 tera wavlm; do
for pretrain_model in whisper_tiny; do
    for dataset in iemocap crema_d msp-improv; do
        CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 gender_inference.py --pretrain_model $pretrain_model --dataset $dataset --learning_rate 0.0005 --downstream_model cnn --num_epochs 10 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128 --privacy_attack gender
    done
done
