# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model apc --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model whisper_base --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model whisper_tiny --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model whisper_small --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model wav2vec2_0 --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model tera --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model wavlm --dataset msp-improv --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128

# for pretrain_model in apc whisper_base whisper_tiny whisper_small wav2vec2_0 tera wavlm; do
# for pretrain_model in apc; do
for pretrain_model in apc whisper_base whisper_tiny whisper_small wav2vec2_0 tera wavlm; do
    CUDA_VISIBLE_DEVICES=0, taskset -c 90-120 python3 fairness_evaluation.py --pretrain_model $pretrain_model --dataset crema_d --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
done