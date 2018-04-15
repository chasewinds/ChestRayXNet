#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

python code/train_muti_label.py \
    --image_set_dir '/comvol/nfs/datasets/medicine/NIH-CXR8/images' \
    --tfrecord_dir 'data/tfrecord/14muti' \
    --tfrecord_prefix 'chest14_muti' \
    --log_dir 'log/14muti' \
    --image_label_list 'data/list/14muti/label_lesion.txt' \
    --checkpoint_file 'model/inception_resnet_v2_2016_08_30.ckpt' \
    --num_classes 14 \
    --num_epoch 6000 \
    --batch_size 16 \
    --step_size 30 \
    --learning_rate 0.0006 \
    --lr_decay_factor 0.5