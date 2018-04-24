#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python code/write_tfrecord.py \
    --dataset_dir '/home/rubans/dataset/images' \
    --write_dir 'data/tfrecord/pneumonia' \
    --train_list 'data/list/pneumonia/train.txt' \
    --val_list 'data/list/pneumonia/val.txt' \
    --num_shards 3 \
    --random_seed 1314 \
    --tfrecord_filename 'pneumonia'