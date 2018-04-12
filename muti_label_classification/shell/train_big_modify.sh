#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0

python code/train_generate.py \
    --image_set_dir '/workvol/home1/lxz/work/NIH-CHEST-Split/sub-NIH-Chest-X-rays/images-100k' \
    --tfrecord_dir '/workvol/home1/lxz/proj/muti_label_classification/data/tfrecord/bak' \
    --tfrecord_prefix 'chest14' \
    --log_dir './log' \
    --image_label_list 'data/list/pneumonia/image_label.txt' \
    --checkpoint_file 'model/inception_resnet_v2_2016_08_30.ckpt' \
    --num_classes 2 \
    --num_epoch 600 \
    --batch_size 8 \
    --step_size 60 \
    --learning_rate 0.002 \
    --lr_decay_factor 0.5