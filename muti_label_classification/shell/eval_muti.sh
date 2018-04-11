#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

python code/eval_muti.py \
    --log_dir 'log/densenet' \
    --log_eval 'log/log_eval_test' \
    --dataset_dir 'data/tfrecord/all' \
    --auc_picture_path 'eval/auc_path/auc.png' \
    --tfrecord_prefix 'chest14_all' \
    --batch_size 16 \
    --num_epochs 1 \
    --num_classes 14 \
    --ckpt_id 1