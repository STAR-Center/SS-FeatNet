#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#data_dir="./example_data"
data_dir="/home/yijun/Documents/roboGit/3dfnv/data/oxford/test_models"
kp_dir="./kp_data"
#res_dir="./test_results/3dfeatnet"
#test_name="not100lr5jit70epoch"
#test_name="sample"
test_name='attentiveFeat' #3dv_repeat'
res_dir="./test_results/${test_name}" 

cd ..

python inference.py  --model='3DFeatNet5' --data_dir=${data_dir} --use_keypoints_from=${kp_dir} --output_dir=${res_dir} --checkpoint=./ckpt/${test_name}/ckpt/checkpoint.ckpt-64500 
#
#python inference.py  --model='3DFeatNet2' --data_dir=${data_dir} --use_keypoints_from=${kp_dir} --output_dir=${res_dir} --checkpoint=./ckpt/${test_name}/ckpt/checkpoint.ckpt-72500
#python inference.py  --data_dir=${data_dir} --use_keypoints_from=${kp_dir} --output_dir=${res_dir} --checkpoint=./ckpt/${test_name}/ckpt/checkpoint.ckpt
#72500

#python inference.py  --data_dir=${data_dir}  --output_dir=${res_dir} --checkpoint=./ckpt/not100lr3jitAttention/ckpt/checkpoint.ckpt-72500 

cd exp
