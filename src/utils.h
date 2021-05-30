//
// Created by yuwei on 4/28/21.
//

#ifndef RGBD_MAPPING_UTILS_H
#define RGBD_MAPPING_UTILS_H

#endif //RGBD_MAPPING_UTILS_H

#include "params.h"

#include <iostream>
#include <fstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>

typedef pcl::PointXYZRGB PointRGBT;
typedef pcl::PointXYZ PointT;

struct Frame{
    std::vector<cv::KeyPoint> keypoints;
    std::vector<PointT> keypoints3D;
    cv::Mat descriptors;
    Eigen::Isometry3f pose;
    int id;
    int file_id;
    bool is_keyframe = false;
};

PointT pixelTo3DCoord(const cv::Point2f pixel_coord, const cv::Mat& depth_img, bool switch_xy = false);

std::vector<int> sampleSubset(const int& total_size, const int& subset_size);

void createPointCloudFromRGBD(const cv::Mat& rgb_image, const cv::Mat& depth_image, pcl::PointCloud<PointRGBT>::Ptr& output_cloud);

void downSampleCloud(pcl::PointCloud<PointRGBT>& cloud, float leaf_size = 0.05);

template <typename T>
void statisticalFilter(pcl::PointCloud<T>& cloud){
    // compute an average distance to k nearest neighbors for EACH point (di)
    // Assume all di's form a Gaussian distribution with mean and std,
    // remove every point with di > 1.0 * std from the mean
    pcl::StatisticalOutlierRemoval<T> sor;
    sor.setInputCloud(boost::make_shared<pcl::PointCloud<T>>(cloud));

    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);

    pcl::PointCloud<T> temp;
    sor.filter(temp);
    int num_removed = temp.size()-cloud.size();
//    cout<<"num removed  "<< num_removed << std::endl;
    cloud = temp;
}

void getAllFiles(const std::string& path, std::vector<std::string>& file_paths, std::vector<double>& timestamps);

std::vector<std::string> getTimeMatchedFiles(const std::vector<std::string>& files1, const std::vector<std::string>& files2,
                                             const std::vector<double>& ts1, const std::vector<double>& ts2);
//void drawMatches(const cv::Mat& img1, const cv::Mat& img2, const Frame& frame1, const Frame& frame2, const std::vector<cv::DMatch>& matches);
