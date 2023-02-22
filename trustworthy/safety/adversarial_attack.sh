# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 adversarial_attack.py --pretrain_model apc --dataset iemocap --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128 --attack_method fgsm
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 adversarial_attack.py --pretrain_model whisper_base --dataset iemocap --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128 --attack_method guassian_noise
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 adversarial_attack.py --pretrain_model whisper_tiny --dataset iemocap --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128 --attack_method fgsm
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 adversarial_attack.py --pretrain_model whisper_small --dataset iemocap --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128 --attack_method fgsm
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 adversarial_attack.py --pretrain_model wav2vec2_0 --dataset iemocap --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128 --attack_method fgsm
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 adversarial_attack.py --pretrain_model tera --dataset iemocap --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128 --attack_method fgsm
# CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 adversarial_attack.py --pretrain_model wavlm --dataset iemocap --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128 --attack_method fgsm

# for pretrain_model in apc whisper_base whisper_tiny whisper_small wav2vec2_0 tera wavlm; do
# for pretrain_model in apc whisper_base whisper_tiny whisper_small wav2vec2_0 tera wavlm; do
#    for dataset in crema_d ravdess msp-improv; do
#        CUDA_VISIBLE_DEVICES=1, taskset -c 90-120 python3 adversarial_attack.py --pretrain_model $pretrain_model --dataset $dataset --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128 --attack_method guassian_noise
#    done
# done

# for pretrain_model in apc whisper_base whisper_tiny whisper_small wav2vec2_0 tera wavlm; do
for pretrain_model in whisper_tiny whisper_small wav2vec2_0 tera wavlm; do
    # for pretrain_model in wav2vec2_0; do
    for dataset in crema_d iemocap msp-improv; do
        CUDA_VISIBLE_DEVICES=0, taskset -c 90-120 python3 adversarial_attack.py --pretrain_model $pretrain_model --dataset $dataset --learning_rate 0.0005 --downstream_model cnn --num_epochs 30 --num_layers 3 --conv_layers 2 --pooling mean --hidden_size 128 --attack_method fgsm
    done
done
