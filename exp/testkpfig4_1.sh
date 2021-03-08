#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#data_dir="./example_data"
data_dir="/home/yijun/Documents/roboGit/3dfnv/data/oxford/test_models"
kp_dir="./kp_data"
res_dir="./res_dir"

cd ..

cd ${data_dir}
bins=$(ls *.bin)
cd -

for binfile in $bins
do
    inbin=$(echo $binfile | cut -d '.' -f1)
    echo $inbin
    ./cpp/build/kpdetect ${data_dir}/${inbin}.bin ${kp_dir}/${inbin}_kp.bin
done
mkdir $kp_dir
mkdir $res_dir

cd exp
