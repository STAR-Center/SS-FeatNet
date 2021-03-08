addpath('./external');

clear;
rte_list= [];
rre_list= [];
successRate_list=[];
Iter_list=[];

for i = 0:10
    DATA_FOLDER = sprintf('../data/kitti/%02d', i);
    ALGO = sprintf('learnDesc_1key_256/%02d', i);
    
    [rte_, rre_, successRate_, Iter_]=registrate_kitti_func(DATA_FOLDER, ALGO);
    
    rte_list=[rte_list, rte_];
    rre_list=[rre_list,rre_];
    successRate_list=[successRate_list,successRate_];
    Iter_list=[Iter_list,Iter_];
    
end

test_nm=[541, 293, 623, 58, 41, 320, 205, 74, 398, 182, 96];
rte_sum = 0;
rre_sum = 0;
success_sum = 0;
iter_sum = 0;
for i = 1:11
    rte_sum = rte_sum+test_nm(i)*successRate_list(i)*rte_list(i);
    rre_sum = rre_sum+test_nm(i)*successRate_list(i)*rre_list(i);
    success_sum = success_sum + test_nm(i)*successRate_list(i);
    iter_sum = iter_sum + test_nm(i) * Iter_list(i);
end
test_nm_sum = sum(test_nm);
rte_final = rte_sum / success_sum;
rre_final = rre_sum / success_sum;
success_rate_final = success_sum/test_nm_sum;
iter_final = iter_sum/test_nm_sum;