#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python code/train_vgg16.py \
    --image_set_dir '/home/rubans/dataset/images' \
    --tfrecord_dir 'data/tfrecord/all' \
    --tfrecord_prefix 'chest14_all' \
    --log_dir 'log/vgg16' \
    --image_label_list 'data/list/14muti/label_lesion.txt' \
    --model_type 'vgg16' \
    --checkpoint_file 'model/vgg16/vgg_16.ckpt' \
    --num_classes 14 \
    --num_epoch 1000 \
    --batch_size 32 \
    --step_size 50 \
    --learning_rate 0.01 \
    --lr_decay_factor 0.1 \
    --weight_decay 1e-4