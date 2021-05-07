//
// Created by yuwei on 5/4/21.
//


#include "utils.h"
// g2o
#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/distances.h>
#include <pcl/registration/transformation_estimation_3point.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <Eigen/Eigen>
#include <numeric>

#define assertm(exp, msg) assert(((void)msg, exp))

class RgbdSLAM {

    std::vector<std::string> rgb_files_, depth_files_;


    pcl::PointCloud<PointRGBT>::Ptr cloud_accum_;

    pcl::visualization::PCLVisualizer::Ptr viewer_;
    int vp1_, vp2_; //pcl viewports

    cv::Ptr<cv::SiftFeatureDetector> detector_;

    g2o::SparseOptimizer optimizer_;
    g2o::RobustKernel* kernel_ptr_;

    std::vector<Frame> all_frames_;
    std::vector<Frame> keyframes_;

    std::vector<pcl::PointCloud<PointRGBT>::Ptr> all_clouds_;

    Frame prev_frame_;
    Frame curr_frame_;

    cv::Mat prev_rgb_img_;  // for drawing matches
    pcl::PointCloud<PointRGBT>::Ptr prev_cloud_; // keep the previous dense cloud for icp (if enabled)
//    Eigen::Matrix4f global_transform_;
    int frame_index_;
    int frame_file_index_;

public:

    RgbdSLAM(const std::vector<std::string>& rgb_files, const std::vector<std::string>& depth_files);

    void run();

    bool readNextRGBDFrame(cv::Mat& rgb_img, cv::Mat& depth_img);

    void detectFeature(const cv::Mat& rgb_img, const cv::Mat& depth_img, const cv::Ptr<cv::SiftFeatureDetector>& detector, Frame& frame);

    std::vector<cv::DMatch> findMatches(const Frame& frame1, const Frame& frame2);

    std::vector<cv::DMatch> performRANSAC(const Frame& frame1, const Frame& frame2, const std::vector<cv::DMatch>& matches);

    Eigen::Matrix4f estimatePairTransform(const Frame& frame1, const Frame& frame2, const std::vector<cv::DMatch> matches);

    void pairAlignCloud(const pcl::PointCloud<PointRGBT>::Ptr& cloud1,
                        const pcl::PointCloud<PointRGBT>::Ptr& cloud2,
                        pcl::PointCloud<PointRGBT>::Ptr& cloud2_aligned,
                        Eigen::Matrix4f& transform,
                        const bool apply_icp = false);


    bool isNewKeyframe(const Frame& prev_keyframe, const Frame& curr_frame);

    int checkLoopClosure(const Frame& curr_keyframe);

    void addNeighboringConstraints(const Frame& current_frame);

    void optimizePoseGraph();

    void visualizeResultMap();

    void visualizeKeyframeMap();

    void createPointCloudFromFile(const int& file_id, pcl::PointCloud<PointRGBT>::Ptr& output_cloud);

    void drawMatches(const cv::Mat& img1, const cv::Mat& img2, const Frame& frame1, const Frame& frame2, const std::vector<cv::DMatch>& matches);
};


