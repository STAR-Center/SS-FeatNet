#/bin/bash

cd ..

DATASET_DIR=./data/oxford
GPU_ID=0
LOG_DIR=./ckpt

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# train
python testDescriptorMatching_keypoint.py --data_dir=$DATASET_DIR --model='3DFeatNet2' --base_scale=2.0 --gpu=$GPU_ID --checkpoint=$LOG_DIR/not100lr3jit_repeat/ckpt/checkpoint.ckpt-72500
#python testDescriptorMatching_keypoint.py --data_dir=$DATASET_DIR  --base_scale=2.0 --gpu=$GPU_ID --checkpoint=$LOG_DIR/sample/ckpt/checkpoint.ckpt 
  #--checkpoint $LOG_DIR/3dfeatnet/ckpt/checkpoint.ckpt-262000
 #--checkpoint $LOG_DIR/not100lr3jit/ckpt/checkpoint.ckpt-72500  
  #--model '3DFeatNet2' \ 
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
