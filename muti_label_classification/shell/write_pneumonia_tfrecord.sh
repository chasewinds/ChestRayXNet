#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

python code/write_tfrecord.py \
    --dataset_dir '/comvol/nfs/datasets/medicine/NIH-CXR8/images' \
    --write_dir 'data/tfrecord' \
    --train_list 'data/list/pneumonia/train.txt' \
    --val_list 'data/list/pneumonia/val.txt' \
    --num_shards 3 \
    --random_seed 1314 \
    --tfrecord_filename 'chest14'