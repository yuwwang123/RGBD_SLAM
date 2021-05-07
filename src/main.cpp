//
// Created by yuwei on 4/20/21.
//

//#include "pairwise_align.h"
#include "rgbd_slam.h"



using namespace std;

const Parameters params("/home/yuwei/Documents/sensor_fusion_projects/rgbd_mapping/src/config_params.xml");




int main(int argc, const char* argv[])
{
    vector<string> all_rgb_files = getAllFiles(params.data_root+"rgb.txt");
    vector<string> all_depth_files = getAllFiles(params.data_root+"depth.txt");

    vector<string> rgb_files, depth_files;
    int idx = params.start_frame;
    int end = std::min({(int)(all_rgb_files.size()), (int)(all_depth_files.size()), params.end_frame});

    while (idx < end){
        rgb_files.push_back(all_rgb_files[idx]);
        depth_files.push_back(all_depth_files[idx]);
        idx += params.frame_skip;
    }

    RgbdSLAM slam(rgb_files, depth_files);

    for(int i=0; i<rgb_files.size(); ++i){
        slam.run();
    }
    slam.optimizePoseGraph();
//    slam.visualizeResultMap();
    slam.visualizeKeyframeMap();


    return 0;
}