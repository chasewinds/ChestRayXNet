#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

python code/write_tfrecord.py \
    --dataset_dir '/comvol/nfs/datasets/medicine/NIH-CXR8/images/images' \
    --write_dir 'data/tfrecord/effusion' \
    --train_list 'data/list/effusion/train.txt' \
    --val_list 'data/list/effusion/val.txt' \
    --num_shards 5 \
    --random_seed 1234 \
    --tfrecord_filename 'chest14'