#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

python code/train_densenet.py \
    --image_set_dir '/comvol/nfs/datasets/medicine/NIH-CXR8/images' \
    --tfrecord_dir 'data/tfrecord/all' \
    --tfrecord_prefix 'chest14_all' \
    --log_dir 'log/debug' \
    --image_label_list 'data/list/14muti/label_lesion.txt' \
    --num_classes 14 \
    --num_epoch 1000 \
    --batch_size 1 \
    --step_size 50 \
    --learning_rate 0.01 \
    --lr_decay_factor 0.1