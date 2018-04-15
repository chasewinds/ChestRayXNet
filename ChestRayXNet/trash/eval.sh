#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

python code/eval_plot.py \
    --log_dir 'log/pne_log' \
    --log_eval 'log/log_eval_test' \
    --dataset_dir 'data/tfrecord' \
    --auc_picture_path 'eval/auc_path/auc.png' \
    --batch_size 27 \
    --num_epochs 1