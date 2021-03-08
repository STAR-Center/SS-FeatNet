#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#data_dir="./example_data"
kitti_data_dir="/home/yijun/Documents/roboGit/3dfnv/data/kitti/"
kitti_kp_dir="./3dfeatnetkp_kitti_kp_data"
#res_dir="./test_results/3dfeatnet"
test_name="sample"
#test_name="3dfeatnet"
kitti_res_dir="./tmp/${test_name}" 

cd ..

for kitti_id in 00 # 01 02 03 04 05 06 07 08 09 10
do 
    data_dir=${kitti_data_dir}${kitti_id}
    kp_dir=${kitti_kp_dir}/${kitti_id}
    res_dir=${kitti_res_dir}/${kitti_id}

    mkdir -p ${kp_dir}


    #python inference.py  --model='3DFeatNet2' --data_dir=${data_dir} --use_keypoints_from=${kp_dir} --output_dir=${res_dir} --checkpoint=./ckpt/${test_name}/ckpt/checkpoint.ckpt-72500
    python 3dfeatnetkp.py --randomize_points  --data_dir=${data_dir} --use_keypoints_from=${kp_dir} --output_dir=${res_dir} --checkpoint=./ckpt/${test_name}/ckpt/checkpoint.ckpt
done

#72500

#python inference.py  --data_dir=${data_dir}  --output_dir=${res_dir} --checkpoint=./ckpt/not100lr3jitAttention/ckpt/checkpoint.ckpt-72500 

cd exp
