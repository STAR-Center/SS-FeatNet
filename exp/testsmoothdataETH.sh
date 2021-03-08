#/bin/bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

#data_dir="./example_data"
root_dir="/home/yijun/Documents/roboGit/3dfnv/data/ETH/data_bin/"
test_name="not100lr3jit3d"

cd ..
 
for scene in gazebo_summer #kitchen sun3d-home_md-home_md_scan9_2012_sep_30 sun3d-hotel_umd-maryland_hotel1 sun3d-mit_76_studyroom-76-1studyroom2 sun3d-home_at-home_at_scan1_2013_jan_1 sun3d-hotel_uc-scan3 sun3d-hotel_umd-maryland_hotel3 sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika
do 
    data_dir=${root_dir}${scene}/pc/
    kp_dir=${root_dir}${scene}/kp/
    res_dir=${root_dir}${test_name}/${scene}/desc_bin/

    python inference.py  --model='3DfeatNet23D' --base_scale=0.620 --data_dir=${data_dir} --use_keypoints_from=${kp_dir} --output_dir=${res_dir} --checkpoint=./ckpt/${test_name}/ckpt/checkpoint.ckpt-136500
done
#72500

#python inference.py  --data_dir=${data_dir}  --output_dir=${res_dir} --checkpoint=./ckpt/not100lr3jitAttention/ckpt/checkpoint.ckpt-72500 

cd exp
