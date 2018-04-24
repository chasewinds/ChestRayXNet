#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python code/train_generate.py \
    --image_set_dir '/home/rubans/dataset/images' \
    --tfrecord_dir 'data/tfrecord/pneumonia' \
    --tfrecord_prefix 'pneumonia' \
    --log_dir 'log/pne_log' \
    --image_label_list 'data/list/pneumonia/image_label.txt' \
    --checkpoint_file 'model/dense121/tf-densenet121.ckpt' \
    --num_classes 1 \
    --num_epoch 6000 \
    --batch_size 16 \
    --step_size 15 \
    --learning_rate 0.001 \
    --lr_decay_factor 0.1