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


PointT pixelTo3DCoord(const cv::Point2f pixel_coord, const cv::Mat& depth_img, bool switch_xy = false){

    int pixel_row, pixel_col;
    if (switch_xy){
        pixel_row = (int)round(pixel_coord.y);
        pixel_col = (int)round(pixel_coord.x);
    }
    else{
        pixel_row = (int)round(pixel_coord.x);
        pixel_col = (int)round(pixel_coord.y);
    }

    float z = depth_img.at<uint16_t>(pixel_row, pixel_col)/params.factor;
    float y = (pixel_row-params.cx) * z / params.fx;
    float x = (pixel_col-params.cy) * z / params.fy;
    return PointT(x, y, z);
}

template <typename T>
void downSampleCloud(const pcl::PointCloud<T>& cloud_in,  pcl::PointCloud<T>& cloud_out, float leaf_size = 0.05){

    pcl::VoxelGrid<PointRGBT> grid;
    grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    grid.setInputCloud(boost::make_shared<pcl::PointCloud<T>>(cloud_in));
    grid.filter(cloud_out);
}


void createPointCloudFromRGBD(pcl::PointCloud<PointRGBT>::Ptr& output_cloud, const cv::Mat& rgb_image, const cv::Mat& depth_image){
    assert((rgb_image.rows==depth_image.rows) && (rgb_image.cols==depth_image.cols));

    pcl::PointCloud<PointRGBT>::Ptr point_cloud (new pcl::PointCloud<PointRGBT>());

    for (int row=0; row<rgb_image.rows; ++row){
        for (int col=0; col<rgb_image.cols; ++col){
            if (depth_image.at<uint16_t>(row, col) == 0) continue;

            PointT coord = pixelTo3DCoord(cv::Point2f(row, col), depth_image);

            PointRGBT p;
            p.x = coord.x; p.y = coord.y; p.z =  coord.z;
            cv::Vec3b bgr =rgb_image.at<cv::Vec3b>(row, col);
            p.b = bgr[0]; p.g = bgr[1]; p.r = bgr[2];

            point_cloud->push_back(p);
        }
    }

    pcl::PassThrough<PointRGBT> pass_filter;
    pass_filter.setInputCloud(point_cloud);
    pass_filter.setFilterFieldName("z");
    pass_filter.setFilterLimits(params.min_depth, params.max_depth);
    pass_filter.filter(*output_cloud);
}


template <typename T>
void statistical_filter(pcl::PointCloud<T>& cloud){
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


template <typename T>
std::vector<T> sampleSubset(const std::vector<T>& input, const int& subset_size){
    std::vector<int> indices(input.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    std::vector<T> subset;
    for (size_t i=0; i<subset_size; ++i){
        subset.push_back(input[indices[i]]);
    }
    return subset;
}


std::vector<std::string> getAllFiles(const std::string& path){

    std::vector<std::string> file_paths;
    std::ifstream fs(path);
    std::string line;

    while (std::getline(fs, line)){
        if (line[0]=='#') continue;
        std::stringstream ss(line);
        std::string temp, file_name;
        getline(ss, temp, ' ');
        getline(ss, file_name);
        file_paths.push_back(params.data_root+file_name);
    }

    return file_paths;
}
