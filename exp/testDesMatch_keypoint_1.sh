#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#data_dir="./example_data"
data_dir="/home/yijun/Documents/roboGit/3dfnv/data/oxford/clusters"
#test_name="3dfeatnet"
test_name="sample"
kp_dir="./clustersTest_iss_kp_data"
res_dir="tmp"

cd ..

mkdir $kp_dir

cd ${data_dir}
bins=$(ls *.bin)
cd -

for binfile in $bins
do
    inbin=$(echo $binfile | cut -d '.' -f1)
    echo $inbin
    ./cpp/build/kpdetect ${data_dir}/${inbin}.bin ${kp_dir}/${inbin}_kp.bin
done

#python 3dfeatnetkp.py --randomize_points  --data_dir=${data_dir} --use_keypoints_from=${kp_dir} --output_dir=${res_dir} --checkpoint=./ckpt/${test_name}/ckpt/checkpoint.ckpt

cd exp
