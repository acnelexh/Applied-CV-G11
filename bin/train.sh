#!/bin/bash

# This script is used to train the model.
python train.py \
    --content_dir input_content \
    --style_texts input_style/style.txt \
    --source_texts input_style/source.txt \
    --vgg ./experiments/vgg_normalised.pth \
    --save_dir ./experiments \
    --log_dir ./logs \
    --device cuda \
    --lr 1e-3 \
    --lr_decay 1e-5 \
    --max_iter 1000 \
    --save_model_interval 5000 \
    --batch_size 1 \
    --clip_model openai/clip-vit-base-patch16\
    --lambda_tv 0 \
    --lambda_patch 1 \
    --lambda_dir 1 \
    --lambda_c 10 \
    --n_threads 0 \
    --thresh 0.7 \
    --crop_size 128 \
    --num_crops 64 \
    --prompt_engineering True \
    --input_size 224 \
    --encoder_embed_dim 512 \
    --encoder_ffn_dim 256 \
    --encoder_depth 6 \
    --encoder_heads 8 \
    --encoder_dropout 0.1 \
    --encoder_activation relu \
    --encoder_normalize_before True \
    --exp_name content_w_mse_and_patch_dir_loss



