#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python code/14muti_tfrecord.py \
    --dataset_dir '/comvol/nfs/datasets/medicine/NIH-CXR8/images' \
    --write_dir 'data/tfrecord/all' \
    --train_list 'data/list/all/train.txt' \
    --val_list 'data/list/all/val.txt' \
    --test_list 'data/list/all/test.txt' \
    --num_shards 10 \
    --random_seed 1234 \
    --tfrecord_filename 'chest14_all'