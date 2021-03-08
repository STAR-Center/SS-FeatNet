#include<iostream>
#include <fstream> 
#include<pcl/point_types.h>
#include<pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>  
#include<string>
using namespace std;

void  
//loadPLYFile(const char* fileName,  
loadPLYFile(string fileName,   
     pcl::PointCloud<pcl::PointXYZ> &cloud  
)  
{  
  //pcl::PolygonMesh mesh;  
    
  //if ( pcl::io::loadPolygonFile ( fileName, mesh ) == -1 )  
  if(pcl::io::loadPLYFile<pcl::PointXYZ> (fileName, cloud) == -1)
  {  
    PCL_ERROR ( "loadFile faild." );  
    return;  
  }  
  //else  
//pcl::fromPCLPointCloud2<pcl::PointXYZ> ( mesh.cloud, cloud );  
    
  // remove points having values of nan  
//std::vector<int> index;  
  //pcl::removeNaNFromPointCloud ( cloud, cloud, index );  
}  

int main(int argc, char** argv){
  if (argc != 5){
    cout << "Usage: ./ply2bin pointcloud.ply keypointsIndex.txt pointcloud.bin keypoint.bin";
    return 1;
  }
  // load ply
  pcl::PointCloud<pcl::PointXYZ> pc;
  loadPLYFile(argv[1],pc);

  cout << pc.at(0);
  // write cloud to .bin
  ofstream fs(argv[3], ios::binary);
  float pad = 0.;
  for(int i = 0; i < pc.points.size(); i++){
    fs.write((char*) &(pc.points.at(i).x), sizeof(float));
    fs.write((char*) &(pc.points.at(i).y), sizeof(float));
    fs.write((char*) &(pc.points.at(i).z), sizeof(float));
    for(int j = 0; j < 3; j++){
        fs.write((char*)&(pad),sizeof(float));
    }
  }
 

  // load keypoint index and write keypoint to .bin
  ifstream inkfs(argv[2]);
  ofstream fsk(argv[4], ios::binary); 
  unsigned int kp_id = 0;
  while (inkfs.eof()==0){
    inkfs >> kp_id;
    fsk.write((char*) &(pc.points.at(kp_id).x), sizeof(float));
    fsk.write((char*) &(pc.points.at(kp_id).y), sizeof(float));
    fsk.write((char*) &(pc.points.at(kp_id).z), sizeof(float));
  }

 


  return 0;
}
