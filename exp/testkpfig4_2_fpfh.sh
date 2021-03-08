#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#data_dir="./example_data"
data_dir="/home/yijun/Documents/roboGit/3dfnv/data/oxford/test_models"
kp_dir="./kp_data"
test_name="fpfh"
res_dir="./test_results/${test_name}"  

cd ..

cd ${data_dir}
bins=$(ls *.bin)
cd -

mkdir $res_dir

for binfile in $bins
do
    inbin=$(echo $binfile | cut -d '.' -f1)
    echo $inbin
    ./cpp/build/fpfhdesc ${data_dir}/${inbin}.bin ${kp_dir}/${inbin}_kp.bin ${res_dir}/${inbin}.bin
done

cd exp
