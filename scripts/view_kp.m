clear;
m = 6;
DATA_FOLDER = '../data/oxford/test_models/';
RESULT_FOLDER = '../test_results/sample/';
DATA_PAIRS = {'1', '84'};
FEATURE_DIM = 32;
pair = DATA_PAIRS(1,:);
cloud_fnames = {[fullfile(DATA_FOLDER, pair{1}), '.bin'], ...
                [fullfile(DATA_FOLDER, pair{2}), '.bin']};
desc_fnames = {[fullfile(RESULT_FOLDER, pair{1}), '.bin'], ...
                   [fullfile(RESULT_FOLDER, pair{2}), '.bin']};
        
for i = 1 : 1

pointcloud{i} = Utils.loadPointCloud(cloud_fnames{i}, m);

xyz_features = Utils.load_descriptors(desc_fnames{i}, sum(FEATURE_DIM+3));

result{i}.xyz = xyz_features(:, 1:3);
result{i}.desc = xyz_features(:, 4:end);
end
Utils.pcshow_keypoints(pointcloud{1},result{1}.xyz);
view(70,35);