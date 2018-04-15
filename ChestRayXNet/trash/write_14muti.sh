#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python code/14muti_tfrecord.py \
    --dataset_dir '/comvol/nfs/datasets/medicine/NIH-CXR8/images' \
    --write_dir 'data/tfrecord/14muti' \
    --train_list 'data/list/14muti/train.txt' \
    --val_list 'data/list/14muti/val.txt' \
    --num_shards 10 \
    --random_seed 1234 \
    --tfrecord_filename 'chest14_muti'