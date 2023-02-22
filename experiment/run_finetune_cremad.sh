# CUDA_VISIBLE_DEVICES=1 taskset -c 1-30 python3 finetune_single_thread.py --pretrain_model apc --dataset crema_d --learning_rate 0.0005 --downstream_model cnn --num_epochs 20 --num_layers 3 --conv_layers 2 --pooling att --hidden_size 128
# CUDA_VISIBLE_DEVICES=1 taskset -c 1-30 python3 finetune_single_thread.py --pretrain_model apc --dataset crema_d --learning_rate 0.0005 --downstream_model cnn --num_epochs 20 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
# CUDA_VISIBLE_DEVICES=1 taskset -c 1-30 python3 finetune_single_thread.py --pretrain_model tera --dataset crema_d --learning_rate 0.0005 --downstream_model cnn --num_epochs 20 --num_layers 3 --conv_layers 2 --pooling att --hidden_size 128
# CUDA_VISIBLE_DEVICES=1 taskset -c 1-30 python3 finetune_single_thread.py --pretrain_model tera --dataset crema_d --learning_rate 0.0005 --downstream_model cnn --num_epochs 20 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
# CUDA_VISIBLE_DEVICES=1 taskset -c 90-120 python3 finetune_single_thread.py --pretrain_model tera --dataset crema_d --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128
# CUDA_VISIBLE_DEVICES=1 taskset -c 90-120 python3 finetune_single_thread.py --pretrain_model wavlm --dataset crema_d --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128

# export OPENBLAS_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=1 taskset -c 90-120 python3 finetune_single_thread.py --pretrain_model whisper_tiny --dataset crema_d --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 256
# CUDA_VISIBLE_DEVICES=1 taskset -c 90-120 python3 finetune_single_thread.py --pretrain_model whisper_base --dataset crema_d --learning_rate 0.0001 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 256
# CUDA_VISIBLE_DEVICES=1 taskset -c 90-120 python3 finetune_single_thread.py --pretrain_model apc --dataset crema_d --learning_rate 0.0001 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 256

