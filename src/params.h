//
// Created by yuwei on 4/25/21.
//

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

struct Parameters{

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
    const float ransac_dist_thresh_low;
    const float ransac_dist_thresh_high;

    const int min_matches_required;
    const float downsample_grid_size;

    const int icp_max_iter;
    const float icp_max_correspondence_dist;
    const std::string data_root;

    const int num_neighboring_edges;
    const int keyframe_thresh;
    const int num_loop_closure_frames;

    Parameters(const std::string& config_filename);

};

extern const Parameters params;