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
#include <pcl/features/fpfh.h> 
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>

void loadBinFile(std::string filename, pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPtr, int dim=6){
    ifstream fs(filename, ios::binary);
    int length;
    float* buffer;
    fs.seekg (0, ios::end);
    length = fs.tellg()/sizeof(float);
    fs.seekg (0, ios::beg);

    buffer = new float [length];
    fs.read ((char *)buffer, sizeof(float)*length);
    
    fs.close();
    for(int i = 0; i < length/dim; i++){
        pcl::PointXYZ basic_point;
        basic_point.x = buffer[i*dim];
        basic_point.y = buffer[i*dim+1];
        basic_point.z = buffer[i*dim+2];
        cloudPtr->points.push_back(basic_point);
    }
    cloudPtr->width = cloudPtr->points.size ();
    std::cout <<"Original Size: " << cloudPtr->width << std::endl;
    cloudPtr->height = 1;

    delete[] buffer;
}


pcl::PointCloud<pcl::FPFHSignature33>::Ptr computeLocalDescriptors (
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        //constSurfaceNormalsPtr & normals,
        pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints){
        //floatfeature_radius){
        
    // follow http://www.pointclouds.org/assets/iros2011/features.pdf

    //compute normal
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
    normal_estimation.setInputCloud (cloud);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    normal_estimation.setSearchMethod (tree);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_with_normals (new pcl::PointCloud<pcl::Normal>);
    normal_estimation.setRadiusSearch (4);
    //normal_estimation.setKSearch(150);
    normal_estimation.compute (*cloud_with_normals);
     


    //feature for keypoints
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
    //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr ftree (new pcl::KdTreeFLANN<pcl::PointXYZ>); 
    pcl::search::KdTree<pcl::PointXYZ>::Ptr ftree (new pcl::search::KdTree<pcl::PointXYZ>); 
    fpfh_estimation.setSearchMethod (ftree);
    fpfh_estimation.setRadiusSearch (4);
    fpfh_estimation.setSearchSurface (cloud);
    //fpfh_estimation.setKSearch(150);
    fpfh_estimation.setInputNormals (cloud_with_normals);
    fpfh_estimation.setInputCloud (keypoints);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr  local_descriptors (new pcl::PointCloud<pcl::FPFHSignature33>);
    fpfh_estimation.compute (*local_descriptors);
    return local_descriptors;
}





                                                     



int main(int, char** argv){
    /*
     * 1. given pc, which is read from .bin (follow the data format in 3dfeatnet)
     * 2. given the keypoint
     * 3. save xyz+desc to bin file
     */
    std::string infile = std::string(argv[1]);
    std::string kpfile = std::string(argv[2]);
    std::string outfile = std::string(argv[3]);
    pcl::PointCloud<pcl::PointXYZ>::Ptr pc1(new pcl::PointCloud<pcl::PointXYZ>);
    loadBinFile(infile, pc1);
    pcl::PointCloud<pcl::PointXYZ>::Ptr kp1(new pcl::PointCloud<pcl::PointXYZ>);
    loadBinFile(kpfile,kp1,3);

    pcl::PointCloud<pcl::FPFHSignature33>::Ptr desc1 = computeLocalDescriptors(pc1, kp1);



    std::ofstream fs(outfile, ios::binary);

    float pad = 0.;
    for(int i = 0; i < kp1->points.size(); i++){
        fs.write((char*) &(kp1->points.at(i).x), sizeof(float));
        fs.write((char*) &(kp1->points.at(i).y), sizeof(float));
        fs.write((char*) &(kp1->points.at(i).z), sizeof(float));
        for(int j = 0; j < 33; j++){
            fs.write((char*)&(desc1->at(i).histogram[j]),sizeof(float));
        }
        
    }
    return 0;
}
