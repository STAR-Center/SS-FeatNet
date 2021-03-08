#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#data_dir="./example_data"
kitti_dir="/home/yijun/Documents/roboGit/3dfnv/data/kitti/"
kitti_kp_dir="./kitti_kp_data"

cd ..

mkdir $kitti_kp_dir

for kitti_id in 00 01 02 03 04 05 06 07 08 09 10
do
    data_dir=${kitti_dir}$kitti_id
    kp_dir=${kitti_kp_dir}/$kitti_id


    cd ${data_dir}
    bins=$(ls *.bin)
    cd -

    mkdir $kp_dir

    for binfile in $bins
    do
        inbin=$(echo $binfile | cut -d '.' -f1)
        echo $inbin
        ./cpp/build/kpdetect ${data_dir}/${inbin}.bin ${kp_dir}/${inbin}_kp.bin
    done
done

cd exp
