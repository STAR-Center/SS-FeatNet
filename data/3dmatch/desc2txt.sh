desc2txt="python desc2txt.py"

out_folder="./data_bin/"
data_root="./3DMatch/"

model_name="not100lr3jit"

for scene in kitchen sun3d-home_md-home_md_scan9_2012_sep_30 sun3d-hotel_umd-maryland_hotel1 sun3d-mit_76_studyroom-76-1studyroom2 sun3d-home_at-home_at_scan1_2013_jan_1 sun3d-hotel_uc-scan3 sun3d-hotel_umd-maryland_hotel3 sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika
do
    scene_fold=${data_root}${scene}/
    find ${scene_fold}/*.ply -type f > .tmpbuf
    length=`cat .tmpbuf | wc -l`
 
    mkdir ${out_folder}${model_name}/${scene}/32_dim/

    for i in $(seq 0  $((${length}-1)) )
    do
        echo ${i}
        ${desc2txt} ${out_folder}${model_name}/${scene}/desc_bin/${i}.bin ${out_folder}${model_name}/${scene}/32_dim/${i}_ss.txt
    done
done





