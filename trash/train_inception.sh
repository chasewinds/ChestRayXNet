#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python code/train_muti_label.py \
    --image_set_dir '/home/rubans/dataset/images' \
    --tfrecord_dir 'data/tfrecord/all' \
    --tfrecord_prefix 'chest14_all' \
    --log_dir 'log/inception' \
    --image_label_list 'data/list/muti_class/label_lesion.txt' \
    --checkpoint_file 'model/inception_resnet_v2_2016_08_30.ckpt' \
    --num_classes 14 \
    --num_epoch 600 \
    --batch_size 16 \
    --step_size 100 \
    --learning_rate 0.0001 \
    --lr_decay_factor 0.7 \
    --weight_decay 1e-4