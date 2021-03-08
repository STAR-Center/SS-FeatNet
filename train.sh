#/bin/bash

DATASET_DIR=/p300/dataset/data/oxford/
LOG_DIR=/p300/dataset/data/oxford/ckptz
GPU_ID=2

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# train
python train1jit.py \
  --model 3DFeatNet2\
  --data_dir $DATASET_DIR \
  --log_dir $LOG_DIR/not100lr3jit_repeat \
  --augmentation Shift \
  --noattention \
  --num_epochs 20 \
  --gpu $GPU_ID
