clear;
model1 = Utils.loadPointCloud('../data/oxford/test_models/0.bin');
%model1 = Utils.loadPointCloud('../data/kitti/00/002652.bin');
figure(1), hold off
pcshow(model1(:, 1:3), 'MarkerSize', 2)
%title('Kitti Data Used')
%xlabel('X/m');
%ylabel('Y/m');
%zlabel('Z/m');