#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#data_dir="./example_data"
data_dir="/home/yijun/Documents/roboGit/3dfnv/data/oxford/clusters_pckp"
pc_dir=${data_dir}/pc
kp_dir=${data_dir}/kp
test_name="fpfh"
res_dir=${data_dir}/desc

cd ..

cd ${kp_dir}
bins=$(ls *.bin)
cd -

mkdir $res_dir

for binfile in $bins
do
    inbin=$(echo $binfile | cut -d '_' -f1)
    echo $inbin
    ./cpp/build/fpfhdesc ${pc_dir}/${inbin}_0.bin ${kp_dir}/${inbin}_kp.bin ${res_dir}/${inbin}_0.bin
    ./cpp/build/fpfhdesc ${pc_dir}/${inbin}_1.bin ${kp_dir}/${inbin}_kp.bin ${res_dir}/${inbin}_1.bin
done

cd exp
