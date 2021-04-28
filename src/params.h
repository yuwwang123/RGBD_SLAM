//
// Created by yuwei on 4/25/21.
//

#include <iostream>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

//namespace params{
//
//    const float fx = 525.0; // focal length x
//    const float fy = 525.0; // focal length y
//    const float cx = 319.5;  // optical center x
//    const float cy = 239.5;  // optical center y
//    const float factor = 5000.0; // for the 16-bit PNG files
//
//    const float min_depth = 0.6;
//    const float max_depth = 4.0;
//
//    const int frame_skip = 1;
//    const int start_frame = 225;
//    const int end_frame = 300;
//
//    const int ransac_max_iter = 30;
//    const float ransac_min_dist_thresh = 0.04;
//    const float ransac_max_dist_thresh = 0.12;
//
//    const int min_matches_required = 50;
//    const float downsample_grid_size = 0.01;
//
//    const int icp_max_iter = 20;
//    const int icp_max_correspondence_dist = 0.2;
//    const std::string data_root = "/home/yuwei/Documents/sensor_fusion_projects/rgbd_mapping/data/rgbd_dataset_freiburg1_room/";
//}


struct Parameters{
//    const float fx = 525.0; // focal length x
//    const float fy = 525.0; // focal length y
//    const float cx = 319.5;  // optical center x
//    const float cy = 239.5;  // optical center y
//    const float factor = 5000.0; // for the 16-bit PNG files
//
//    const float min_depth = 0.6;
//    const float max_depth = 4.0;
//
//    const int frame_skip = 1;
//    const int start_frame = 225;
//    const int end_frame = 300;
//
//    const int ransac_max_iter = 30;
//    const float ransac_min_dist_thresh = 0.04;
//    const float ransac_max_dist_thresh = 0.12;
//
//    const int min_matches_required = 50;
//    const float downsample_grid_size = 0.01;
//
//    const int icp_max_iter = 20;
//    const int icp_max_correspondence_dist = 0.2;
//    const std::string data_root = "/home/yuwei/Documents/sensor_fusion_projects/rgbd_mapping/data/rgbd_dataset_freiburg1_room/";
    const float fx; // focal length x
    const float fy; // focal length y
    const float cx;  // optical center x
    const float cy;  // optical center y
    const float factor; // for the 16-bit PNG files

    const float min_depth;
    const float max_depth;

    const int run_mode;
    const int frame_skip;
    const int start_frame;
    const int end_frame;

    const int ransac_max_iter;
    const float ransac_min_dist_thresh;
    const float ransac_max_dist_thresh;

    const int min_matches_required;
    const float downsample_grid_size;

    const int icp_max_iter;
    const float icp_max_correspondence_dist;
    const std::string data_root;

    Parameters(const std::string& config_filename);

};

extern const Parameters params;