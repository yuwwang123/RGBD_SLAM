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


std::vector<int> sampleSubset(const int& total_size, const int& subset_size){
    std::vector<int> indices(total_size);
    std::iota(indices.begin(), indices.end(), 0);

    if (total_size <= subset_size) {
        return indices;
    }

    std::random_shuffle(indices.begin(), indices.end());

    std::vector<int> subset(indices.begin(), indices.begin()+subset_size);
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



void getAllFiles(const std::string& path, std::vector<std::string>& file_paths, std::vector<double>& timestamps){

    std::ifstream fs(path);
    std::string line;

    while (std::getline(fs, line)){
        if (line[0]=='#') continue;
        std::stringstream ss(line);
        std::string timestamp_str, file_name;
        getline(ss, timestamp_str, ' ');
        getline(ss, file_name);
        file_paths.push_back(params.data_root+file_name);
        timestamps.push_back(std::stod(timestamp_str));
    }

}

std::vector<std::string> getTimeMatchedFiles(const std::vector<std::string>& files1, const std::vector<std::string>& files2,
                     const std::vector<double>& ts1, const std::vector<double>& ts2){
    assert(files1.size()<=files2.size());
    std::vector<std::string> files2_new;

    for (double t : ts1){
        auto it = std::min_element(ts2.begin(),
                                   ts2.end(),
                                   [t](double a, double b){ return std::abs(t-a) < std::abs(t-b);});

        std::cout<<t/1e9<<"   "<<*it/1e9<<std::endl;
        files2_new.push_back(files2[std::distance(ts2.begin(), it)]);
    }

    assert(files1.size() == files2_new.size());

    for (int i=0; i<files1.size(); i++){
        std::cout<<files1[i]<<"    "<<files2_new[i]<<std::endl;
    }

    return files2_new;
}