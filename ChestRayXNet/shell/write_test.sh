#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python code/test_tfrecord.py \
    --dataset_dir '/comvol/nfs/datasets/medicine/NIH-CXR8/images' \
    --write_dir 'data/tfrecord/4muti' \
    --test_list 'data/list/4muti/test.txt' \
    --num_shards 5 \
    --random_seed 1234 \
    --tfrecord_filename 'chest14_muti'