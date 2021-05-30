//
// Created by yuwei on 4/20/21.
//

//#include "pairwise_align.h"
#include "rgbd_slam.h"



using namespace std;

const Parameters params("/home/yuwei/Documents/sensor_fusion_projects/rgbd_mapping/src/config_params.xml");




int main(int argc, const char* argv[])
{
    vector<string> all_rgb_files, all_depth_files;
    vector<double> rgb_ts, depth_ts;

    getAllFiles(params.data_root+"rgb.txt", all_rgb_files, rgb_ts);
    getAllFiles(params.data_root+"depth.txt", all_depth_files, depth_ts);

    if (all_rgb_files.size() < all_depth_files.size()){
        all_depth_files = getTimeMatchedFiles(all_rgb_files, all_depth_files, rgb_ts, depth_ts);
    }
    else{
        all_rgb_files = getTimeMatchedFiles(all_depth_files, all_rgb_files, depth_ts, rgb_ts);
    }

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
//        slam.visualizeResultMap();
    }
    slam.optimizePoseGraph();
//    slam.visualizeResultMap();
    slam.visualizeKeyframeMap();
//    slam.visualizeWholeMap();


    return 0;
}