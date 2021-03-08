#include <string>
#include <fstream>
#include<eigen3/Eigen/Dense>
#include <pcl/common/transforms.h>
#include <pcl/features/fpfh.h>
#include <pcl/io/pcd_io.h> 
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <thread>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/random_sample.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/iss_3d.h>

void loadBinFile(std::string filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr){
    ifstream fs(filename, ios::binary);
    int length;
    float* buffer;
    fs.seekg (0, ios::end);
    length = fs.tellg()/sizeof(float);
    fs.seekg (0, ios::beg);

    buffer = new float [length];
    fs.read ((char *)buffer, sizeof(float)*length);
    
    fs.close();
    for(int i = 0; i < length/6; i++){
        pcl::PointXYZ basic_point;
        basic_point.x = buffer[i*6];
        basic_point.y = buffer[i*6+1];
        basic_point.z = buffer[i*6+2];
        cloudPtr->points.push_back(basic_point);
    }
    cloudPtr->width = cloudPtr->points.size ();
    std::cout <<"Original Size: " << cloudPtr->width << std::endl;
    cloudPtr->height = 1;

    delete[] buffer;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr getISSKeyPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr model){ 

    pcl::PointCloud<pcl::PointXYZ>::Ptr model_keypoints (new pcl::PointCloud<pcl::PointXYZ> ());
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());

    pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss_detector;
    iss_detector.setSearchMethod (tree);
    iss_detector.setSalientRadius (1.);
    iss_detector.setNonMaxRadius (0.5);
    iss_detector.setThreshold21 (0.975);
    iss_detector.setThreshold32 (0.975);
    iss_detector.setMinNeighbors (7);

    iss_detector.setInputCloud (model);
    iss_detector.compute (*model_keypoints);

    cout << "Keypoints: " << model_keypoints->width << endl;

    return model_keypoints;
}


 

pcl::PointCloud<pcl::PointXYZ>::Ptr getKeyPointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud){ 
    /* for lounge data
    const float min_scale = 0.01;//50;
    const int nr_octaves = 3;
    const int nr_scales_per_octave = 4;
    const float min_contrast = 0.001f; 
     */
    /* for wuzberg data
    const float min_scale = 50;//50;
    const int nr_octaves = 3;
    const int nr_scales_per_octave = 4;
    const float min_contrast = 0.001f;
     */

    /* for kitti data 
     */
    const float min_scale = 1.;
    const int nr_octaves = 3;
    const int nr_scales_per_octave = 4;
    const float min_contrast = 0.001f; 




    std::vector<Eigen::MatrixXf> diff_gauss_vec;

    // Estimate the normals of the cloud_xyz
    pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals (new pcl::PointCloud<pcl::PointNormal>);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZ>());

    ne.setInputCloud(cloud);
    ne.setSearchMethod(tree_n);
    //ne.setRadiusSearch(5);//0.2);
    ne.setKSearch(150); 
    ne.compute(*cloud_normals);


    // Copy the xyz info from cloud_xyz and add it to cloud_normals as the xyz field in PointNormals estimation is zero
    for(std::size_t i = 0; i<cloud_normals->points.size(); ++i)
    {
    cloud_normals->points[i].x = cloud->points[i].x;
    cloud_normals->points[i].y = cloud->points[i].y;
    cloud_normals->points[i].z = cloud->points[i].z;
    }

    // Estimate the sift interest points using normals values from xyz as the Intensity variants
    pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale> result;
    pcl::search::KdTree<pcl::PointNormal>::Ptr ntree(new pcl::search::KdTree<pcl::PointNormal> ());
    sift.setSearchMethod(ntree);
    sift.setScales(min_scale, nr_octaves, nr_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(cloud_normals);
    sift.compute(result);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp (new pcl::PointCloud<pcl::PointXYZ>);
    copyPointCloud(result, *cloud_temp);

    cout << "Keypoints: " << cloud_temp->width << endl;

    return cloud_temp;
    
}







                                                     



int main(int, char** argv){
    /*
     * 1. given pc, which is read from .bin (follow the data format in 3dfeatnet)
     * 2. detect the keypoint
     * 3. save keypoints to bin file
     */
    std::string infile = std::string(argv[1]);
    std::string outfile = std::string(argv[2]);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc1(new pcl::PointCloud<pcl::PointXYZ>);
    loadBinFile(infile, pc1);

    pcl::PointCloud<pcl::PointXYZ>::Ptr kp1 = getISSKeyPointCloud(pc1);

    std::ofstream fs(outfile, ios::binary);

    float pad = 0.;
    for(int i = 0; i < kp1->points.size(); i++){
	fs.write((char*) &(kp1->points.at(i).x), sizeof(float));
	fs.write((char*) &(kp1->points.at(i).y), sizeof(float));
	fs.write((char*) &(kp1->points.at(i).z), sizeof(float));
	//fs.write((char*) &(pad), sizeof(float));
	//fs.write((char*) &(pad), sizeof(float));
	//fs.write((char*) &(pad), sizeof(float));
    }
    return 0;
}
