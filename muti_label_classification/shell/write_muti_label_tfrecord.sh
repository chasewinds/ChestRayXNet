#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

python code/write_tfrecord.py \
    --dataset_dir '/comvol/nfs/datasets/medicine/NIH-CXR8/images' \
    --write_dir 'data/tfrecord/muti_label' \
    --train_list 'data/list/muti_class/train.txt' \
    --val_list 'data/list/muti_class/val.txt' \
    --num_shards 10 \
    --random_seed 1234 \
    --tfrecord_filename 'chest14'