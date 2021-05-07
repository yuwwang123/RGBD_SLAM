//
// Created by yuwei on 4/27/21.
//

#include "params.h"

using namespace boost::property_tree;

template <class T>
T getValueFromFile(const std::string& filename, const std::string& key){
    ptree tree;
    // Parse the XML into the property tree.
    xml_parser::read_xml(filename, tree);

    return tree.get<T>(key);
}

Parameters::Parameters(const std::string& config_filename) :
        fx(getValueFromFile<float>(config_filename, "params.camera.fx")),
        fy(getValueFromFile<float>(config_filename, "params.camera.fy")),
        cx(getValueFromFile<float>(config_filename, "params.camera.cx")),
        cy(getValueFromFile<float>(config_filename, "params.camera.cy")),
        factor(getValueFromFile<float>(config_filename, "params.camera.factor")),
        min_depth(getValueFromFile<float>(config_filename, "params.camera.min_depth")),
        max_depth(getValueFromFile<float>(config_filename, "params.camera.max_depth")),

        run_mode(getValueFromFile<int>(config_filename, "params.config.run_mode")),

        frame_skip(getValueFromFile<int>(config_filename, "params.config.frame_skip")),
        start_frame(getValueFromFile<int>(config_filename, "params.config.start_frame")),
        end_frame(getValueFromFile<int>(config_filename, "params.config.end_frame")),
        min_matches_required(getValueFromFile<int>(config_filename, "params.config.min_matches_required")),
        downsample_grid_size(getValueFromFile<float>(config_filename, "params.config.downsample_grid_size")),
        data_root(getValueFromFile<std::string>(config_filename, "params.config.data_root")),

        ransac_max_iter(getValueFromFile<int>(config_filename, "params.ransac.max_iter")),
        ransac_dist_thresh_low(getValueFromFile<float>(config_filename, "params.ransac.dist_thresh_low")),
        ransac_dist_thresh_high(getValueFromFile<float>(config_filename, "params.ransac.dist_thresh_high")),

        icp_max_iter(getValueFromFile<int>(config_filename, "params.ransac.max_iter")),
        icp_max_correspondence_dist(getValueFromFile<float>(config_filename, "params.icp.max_correspondence_dist")),

        num_neighboring_edges(getValueFromFile<int>(config_filename, "params.SLAM.num_neighboring_edges")),
        keyframe_thresh(getValueFromFile<int>(config_filename, "params.SLAM.keyframe_thresh")),
        num_loop_closure_frames(getValueFromFile<int>(config_filename, "params.SLAM.num_loop_closure_frames"))
{}