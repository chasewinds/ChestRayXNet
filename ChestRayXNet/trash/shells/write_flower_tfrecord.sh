#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python transfer_learning_tutorial/write_tfrecord.py \
    --dataset_dir 'transfer_learning_tutorial/data/images' \
    --write_dir 'transfer_learning_tutorial/data/tfrecord' \
    --validation_size 0.3 \
    --num_shards 2 \
    --random_seed 1234 \
    --tfrecord_filename 'flowers'