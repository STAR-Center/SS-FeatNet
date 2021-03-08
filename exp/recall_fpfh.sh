#/bin/bash 
cd ..
DATASET_DIR=./data/oxford 
python genCluster.py --data_dir=$DATASET_DIR --recall=1

cd exp
