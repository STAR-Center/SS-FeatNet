% Evaluates descriptor + detector.
% Assumes computed descriptors are stored under '../../results/oxford/f3d_d32'

function [rte_, rre_, successRate_, Iter_] = registrate_kitti_func(DATA_FOLDER, ALGO)
m = 6;  % Dimensionality of raw data

%DATA_FOLDER = '../data/oxford/test_models';
%RESULT_FOLDER = '../test_results';

%DATA_FOLDER = '../data/kitti/10';
RESULT_FOLDER = '../kitti_test_results';

%ALGO = '3dfeatnet/10'; 
featureDim = 32;
%ALGO = 'fpfh'; featureDim = 33;
%% Load pairs
algoResultFolder = fullfile(RESULT_FOLDER, ALGO);
test_pairs = readtable(fullfile(DATA_FOLDER, 'groundtruths.txt'));

statistic.idx = [];
statistic.RTE = [];
statistic.RRE = [];
statistic.success = [];
statistic.iters = []; 

tic
for iPair = 1 : height(test_pairs)
    
    frames = [test_pairs.idx1(iPair), test_pairs.idx2(iPair)];
    fprintf('Running pair %i of %i, containing frames %i and %i\n', ...
        iPair, height(test_pairs), frames(1), frames(2));
    
    % Load point cloud and descriptors
    for i = 1 : 2
        r = frames(i);

        %pointcloud{i} = Utils.loadPointCloud(fullfile(DATA_FOLDER, sprintf('%d.bin', r)), m);

        binfile = fullfile(algoResultFolder, sprintf('%06d.bin', r));
        %binfile = fullfile(algoResultFolder, sprintf('%06d.bin', r));
        xyz_features = Utils.load_descriptors(binfile, sum(featureDim+3));

        result{i}.xyz = xyz_features(:, 1:3);
        result{i}.desc = xyz_features(:, 4:end);
    end

    % Match
    [~, matches12] = pdist2(result{2}.desc, result{1}.desc, 'euclidean', 'smallest', 1);
    matches12 = [1:length(matches12); matches12]';  

    %  RANSAC
    cloud1_pts = result{1}.xyz(matches12(:,1), :);
    cloud2_pts = result{2}.xyz(matches12(:,2), :);
    [estimateRt, inlierIdx, trialCount] = ransacfitRt([cloud1_pts'; cloud2_pts'], 1.0, false);
    fprintf('Number of inliers: %i / %i (Proportion: %.3f. #RANSAC trials: %i)\n', ...
            length(inlierIdx), size(matches12, 1), ...
            length(inlierIdx)/size(matches12, 1), trialCount);
    
    % Load Groundtruth
    t_gt = [test_pairs.t_1(iPair), test_pairs.t_2(iPair), test_pairs.t_3(iPair)];
    q_gt = [test_pairs.q_1(iPair), test_pairs.q_2(iPair), test_pairs.q_3(iPair), test_pairs.q_4(iPair)];
    R_gt = quat2rotm(q_gt);
    %T_gt = [quat2rotm(q_gt) t_gt'; 0 0 0 1];

    % RTE and RRE
    estimateR = estimateRt(1:3,1:3);
    estimatet = estimateRt(1:3,4)';

    rte = sqrt(sum((estimatet-t_gt).^2));
    rre = sum(abs(rotm2eul(inv(R_gt)*estimateR)./pi*180));

    if (rte < 2. & rre < 5.)
        success = 1.;
        statistic.RTE = [statistic.RTE, rte];
        statistic.RRE = [statistic.RRE, rre];
    else
        success = 0.;
    end

    statistic.idx = [statistic.idx, iPair];
    statistic.success = [statistic.success, success];
    statistic.iters = [statistic.iters, trialCount];
end

fprintf('RTE: %f, RRE: %f, SuccessRate: %f, AvgIter: %f',...
        mean(statistic.RTE), mean(statistic.RRE), mean(statistic.success), mean(statistic.iters));
    rte_=mean(statistic.RTE);
    rre_=mean(statistic.RRE);
    successRate_ = mean(statistic.success);
    Iter_=mean(statistic.iters);
end

