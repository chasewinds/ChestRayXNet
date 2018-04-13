#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python code/train_densenet.py \
    --image_set_dir '/comvol/nfs/datasets/medicine/NIH-CXR8/images' \
    --tfrecord_dir 'data/tfrecord/all' \
    --tfrecord_prefix 'chest14_all' \
    --log_dir 'log/densenet_debug' \
    --image_label_list 'data/list/14muti/label_lesion.txt' \
    --checkpoint_file 'model/densenet/tf-densenet161.ckpt' \
    --num_classes 14 \
    --num_epoch 1000 \
    --batch_size 16 \
    --step_size 50 \
    --learning_rate 0.01 \
    --lr_decay_factor 0.1 \
    --weight_decay 1e-4