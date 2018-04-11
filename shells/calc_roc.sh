#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1
# model_folder=experiments/vgg_ft_single
model_folder = vgg/models/output/pneumonia
experiment=pneumonia
iter=24450
# effusion_iter_9750.caffemodel
python ./vgg/code/test_roc.py \
	-m vgg/net/effusion/deploy_vgg_ft.prototxt \
	-w vgg/models/output/pneumonia/pneumonia_iter_219060.caffemodel \
	-t VGG \
	-b 128 \
	-l vgg/data/list/pneumonia/test.txt \
    -s vgg/models/output/pneumonia/results/scores-pneumonia.csv \
    -r vgg/models/output/pneumonia/results/roc-pneumonia.png \
	-n "Normal vs Abnormal"



#	-s ${model_folde/results/scores-${experiment}.csv \
#	-r ${model_folder}/results/roc-${experiment}.png \

#
#
##!/usr/bin/env bash
#
#model_folder=experiments/vgg_ft_single
#experiment=0329-1-normal-abnormal
#iter=2301
#
#python ./scripts/test_roc.py \
#	-m ${model_folder}/deploy_vgg_ft.prototxt \
#	-t VGG	\
#	-w ${model_folder}/best/${experiment}/${experiment}_iter_${iter}.caffemodel \
#	-l data/${experiment}/val.txt \
#	-s ${model_folder}/results/scores-${experiment}.csv \
#	-r ${model_folder}/results/roc-${experiment}.png \
#	-n "Normal vs Abnormal"

