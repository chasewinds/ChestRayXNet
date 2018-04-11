#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1

python code/train_muti_label.py \
    --image_set_dir '/comvol/nfs/datasets/medicine/NIH-CXR8/images' \
    --tfrecord_dir 'data/tfrecord/4muti' \
    --tfrecord_prefix 'chest14_muti' \
    --log_dir 'log/4muti' \
    --image_label_list 'data/list/4muti/label_lesion.txt' \
    --checkpoint_file 'log/no_l2_4muti/model.ckpt-87441' \
    --num_classes 4 \
    --num_epoch 6000 \
    --batch_size 16 \
    --step_size 10 \
    --learning_rate 0.001 \
    --lr_decay_factor 0.8 \
    --weight_decay 0.00004