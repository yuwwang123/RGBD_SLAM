//
// Created by yuwei on 5/4/21.
//

//
// Created by yuwei on 4/28/21.
//
#include "utils.h"


PointT pixelTo3DCoord(const cv::Point2f pixel_coord, const cv::Mat& depth_img, bool switch_xy){

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


//template <typename T>
std::vector<cv::DMatch> sampleSubset(const std::vector<cv::DMatch>& input, const int& subset_size){
    std::vector<int> indices(input.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    std::vector<cv::DMatch> subset;
    for (size_t i=0; i<subset_size; ++i){
        subset.push_back(input[indices[i]]);
    }
    return subset;
}

void createPointCloudFromRGBD(const cv::Mat& rgb_image, const cv::Mat& depth_image, pcl::PointCloud<PointRGBT>::Ptr& output_cloud){
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


void downSampleCloud(pcl::PointCloud<PointRGBT>& cloud, float leaf_size){

    if (leaf_size < 0.001) return;

    pcl::VoxelGrid<PointRGBT> grid;
    grid.setLeafSize(leaf_size, leaf_size, leaf_size);
    grid.setInputCloud(boost::make_shared<pcl::PointCloud<PointRGBT>>(cloud));

    pcl::PointCloud<PointRGBT> temp;
    grid.filter(temp);
    cloud = temp;
}


//template <typename T>
void statisticalFilter(pcl::PointCloud<PointRGBT>& cloud){
    // compute an average distance to k nearest neighbors for EACH point (di)
    // Assume all di's form a Gaussian distribution with mean and std,
    // remove every point with di > 1.0 * std from the mean
    pcl::StatisticalOutlierRemoval<PointRGBT> sor;
    sor.setInputCloud(boost::make_shared<pcl::PointCloud<PointRGBT>>(cloud));

    sor.setMeanK(50);
    sor.setStddevMulThresh(1.0);

    pcl::PointCloud<PointRGBT> temp;
    sor.filter(temp);
//    int num_removed = temp.size()-cloud.size();
//    cout<<"num removed  "<< num_removed << std::endl;
    cloud = temp;
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

