ply2bin=./build/ply2bin

out_folder="./data_bin/"
mkdir ${out_folder}
data_root="./3DMatch/"
for scene in kitchen sun3d-home_md-home_md_scan9_2012_sep_30 sun3d-hotel_umd-maryland_hotel1 sun3d-mit_76_studyroom-76-1studyroom2 sun3d-home_at-home_at_scan1_2013_jan_1 sun3d-hotel_uc-scan3 sun3d-hotel_umd-maryland_hotel3 sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika
do
    scene_fold=${data_root}${scene}/
    find ${scene_fold}/*.ply -type f > .tmpbuf
    length=`cat .tmpbuf | wc -l`

    mkdir ${out_folder}${scene}
    mkdir ${out_folder}${scene}/pc
    mkdir ${out_folder}${scene}/kp

    echo ${length}
    for i in $(seq 0 $((${length}-1)))
    do
        echo ${scene_fold}cloud_bin_${i}.ply
        ${ply2bin} ${scene_fold}cloud_bin_${i}.ply ${scene_fold}01_Keypoints/cloud_bin_${i}Keypoints.txt  ${out_folder}${scene}/pc/${i}.bin ${out_folder}${scene}/kp/${i}_kp.bin
    done
done





