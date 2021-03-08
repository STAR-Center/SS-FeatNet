#/bin/bash 
cd ..
DATASET_DIR=./data/oxford 
python genCluster.py --data_dir=$DATASET_DIR --recal=1

cd exp
