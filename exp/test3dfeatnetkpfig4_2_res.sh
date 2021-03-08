#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#data_dir="./example_data"
data_dir="/home/yijun/Documents/roboGit/3dfnv/data/oxford/test_models"
kp_dir="./3dfeatnetkp_kp_data"
#res_dir="./test_results/3dfeatnet"
#test_name="sample"
#test_name="not100lr3jit"
#test_name='3dv_repeat'
test_name='resi_net4'
res_dir="./3dfeatnetkp_test_results/${test_name}" 

cd ..

#python inference.py  --model='3DFeatNet2' --data_dir=${data_dir} --use_keypoints_from=${kp_dir} --output_dir=${res_dir} --checkpoint=./ckpt/${test_name}/ckpt/checkpoint.ckpt-72500 
# feat3net3
python inference.py  --model='3DFeatNet4' --data_dir=${data_dir} --use_keypoints_from=${kp_dir} --output_dir=${res_dir} --checkpoint=./ckpt/${test_name}/ckpt/checkpoint.ckpt-72500 
#
#python inference.py  --data_dir=${data_dir} --use_keypoints_from=${kp_dir} --output_dir=${res_dir} --checkpoint=./ckpt/${test_name}/ckpt/checkpoint.ckpt


cd exp
