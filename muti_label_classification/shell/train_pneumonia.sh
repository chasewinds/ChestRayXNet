#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

python code/train_generate.py \
    --image_set_dir '/comvol/nfs/datasets/medicine/NIH-CXR8/images' \
    --tfrecord_dir 'data/tfrecord' \
    --tfrecord_prefix 'chest14' \
    --log_dir 'log/pne_log' \
    --image_label_list 'data/list/pneumonia/image_label.txt' \
    --checkpoint_file 'model/inception_resnet_v2_2016_08_30.ckpt' \
    --num_classes 2 \
    --num_epoch 6000 \
    --batch_size 16 \
    --step_size 100 \
    --learning_rate 0.0004 \
    --lr_decay_factor 0.9