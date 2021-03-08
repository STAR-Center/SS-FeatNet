#/bin/bash

cd ..

DATASET_DIR=./data/oxford
GPU_ID=0
LOG_DIR=./ckpt

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# train
python inference.py \
  --data_dir $DATASET_DIR/test_models \
  --output_dir=./test_results/pretrain72500 \
  --checkpoint $LOG_DIR/pretrain/ckpt/checkpoint.ckpt-72500

# Pretrain
#python train.py \
#  --data_dir $DATASET_DIR \
#  --log_dir $LOG_DIR/pretrain \
#  --augmentation Jitter RotateSmall Shift \
#  --noattention --noregress \
#  --num_epochs 2 \
#  --gpu $GPU_ID
# Second stage training: Performance should saturate in ~60 epochs
#python train.py \
#  --data_dir $DATASET_DIR \
#  --log_dir $LOG_DIR/secondstage \
#  --checkpoint $LOG_DIR/pretrain/ckpt \
#  --restore_exclude detection \
#  --augmentation Jitter RotateSmall Shift Rotate1D \
#  --num_epochs 70 \
#  --gpu $GPU_ID


cd exp
