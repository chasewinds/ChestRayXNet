export CUDA_VISIBLE_DEVICES=0

python scripts/run_chexnet.py \
    --train \
    --model_type='DenseNet-BC' \
    --growth_rate=12 \
    --depth=40 \
    --train_tfname data/tfrecords/train \
    --validate_tfname data/tfrecords/val \
    --total_blocks=4 \
    --keep_prob=1.0 \
    --weight_decay=1e-4 \
    --nesterov_momentum=0.999 \
    --reduction=0.5 \
    --logs \
    --saves \
    --renew-logs