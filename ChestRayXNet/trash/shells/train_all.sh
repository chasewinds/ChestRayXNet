#!/bin/bash
export CUDA_VISIBLE_DEVICES=1
TOOLS=~/files/caffe/build/tools
model_name=vgg_ft_single
weightfile=vgg/models/VGG_ILSVRC_16_layers.caffemodel

datetime=$(date +"%Y%m%d_%H%M%S")
log_file=log/${datetime}_${model_name}.txt
echo ${log_file}

touch ${log_file}
#setsid $TOOLS/caffe train \
#    --solver=/workvol/home1/lyt/proj/luoyt/Xvision/myexperiment/solver_Xvision.prototxt \
#    --weights=${weightfile} \
#    -gpu 0 \
#    >${log_file} \
#    2>&1 &
# tail -f ${log_file}

$TOOLS/caffe train \
    --solver=vgg/net/14_class/solver_vgg_ft.prototxt \
    --weights=${weightfile} \
    -gpu 0

