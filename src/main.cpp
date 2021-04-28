//
// Created by yuwei on 4/20/21.
//

#include "pairwise_align.h"
#include <pcl/visualization/pcl_visualizer.h>


typedef pcl::PointXYZRGB PointRGBT;
typedef pcl::PointXYZ PointT;

using namespace std;
using namespace cv;

const Parameters params("/home/yuwei/Documents/sensor_fusion_projects/rgbd_mapping/src/config_params.xml");


void runSequence(const pcl::visualization::PCLVisualizer::Ptr& viewer,
                 const vector<string>& rgb_files,
                 const vector<string>& depth_files){

    Eigen::Matrix4f global_transform = Eigen::Matrix4f::Identity();
    pcl::PointCloud<PointRGBT>::Ptr cloud_accum(new pcl::PointCloud<PointRGBT>());

    int frame1 = params.start_frame;
    int frame2 = frame1 + params.frame_skip;
    bool firstFrame = true;

    while(frame2 <= params.end_frame){
        cout<<"1st frame: "<<frame1<<", "<< "2nd frame: "<<frame2<<std::endl;

        //Keypoint Detection
        const Mat rgb_img1 = cv::imread(rgb_files[frame1],  cv::ImreadModes::IMREAD_UNCHANGED);
        const Mat depth_img1 = cv::imread(depth_files[frame1],  cv::ImreadModes::IMREAD_UNCHANGED);
        const Mat rgb_img2 = cv::imread(rgb_files[frame2],  cv::ImreadModes::IMREAD_UNCHANGED);
        const Mat depth_img2 = cv::imread(depth_files[frame2],  cv::ImreadModes::IMREAD_UNCHANGED);

        if (firstFrame){
            pcl::PointCloud<PointRGBT>::Ptr temp(new pcl::PointCloud<PointRGBT>());
            createPointCloudFromRGBD(temp, rgb_img1, depth_img1);
            downSampleCloud(*temp, *cloud_accum, params.downsample_grid_size);
            firstFrame = false;
        }

        Eigen::Matrix4f pair_transform;
        pcl::PointCloud<PointRGBT>::Ptr cloud_aligned(new pcl::PointCloud<PointRGBT>());
        pcl::PointCloud<PointRGBT>::Ptr cloud_aligned_global(new pcl::PointCloud<PointRGBT>());
        pcl::PointCloud<PointT> feature_cloud1, feature_cloud2;

        pairAlign(rgb_img1,rgb_img2,
                  depth_img1,depth_img2,
               cloud_aligned,pair_transform,
               feature_cloud1,feature_cloud2);

        // Transformed the cloud back to the first frame's reference frame
        pcl::transformPointCloud(*cloud_aligned, *cloud_aligned_global, global_transform.inverse());
        *cloud_accum += *cloud_aligned_global;
        global_transform = global_transform * pair_transform;

        frame1 = frame2;
        frame2 += params.frame_skip;

        viewer->addPointCloud(cloud_accum, "cloud_accum");
        viewer->spin();
        viewer->removePointCloud("cloud_accum");

    }
    //   pcl::io::savePCDFileASCII ("rgbd_map.pcd", *cloud_accum);
    //    cerr<< "Saved " << cloud_accum->size () << " data points to test_pcd.pcd." << endl;
}


void runPair(const pcl::visualization::PCLVisualizer::Ptr& viewer,
                 const vector<string>& rgb_files,
                 const vector<string>& depth_files,
                 const int& f1_idx,
                 const int& f2_idx){

    //Keypoint Detection
    const Mat rgb_img1 = cv::imread(rgb_files[f1_idx],  cv::ImreadModes::IMREAD_UNCHANGED);
    const Mat depth_img1 = cv::imread(depth_files[f1_idx],  cv::ImreadModes::IMREAD_UNCHANGED);
    const Mat rgb_img2 = cv::imread(rgb_files[f2_idx],  cv::ImreadModes::IMREAD_UNCHANGED);
    const Mat depth_img2 = cv::imread(depth_files[f2_idx],  cv::ImreadModes::IMREAD_UNCHANGED);

    pcl::PointCloud<PointRGBT>::Ptr temp(new pcl::PointCloud<PointRGBT>());
    pcl::PointCloud<PointRGBT>::Ptr cloud_out(new pcl::PointCloud<PointRGBT>());

    createPointCloudFromRGBD(temp, rgb_img1, depth_img1);
    downSampleCloud(*temp, *cloud_out, params.downsample_grid_size);


    Eigen::Matrix4f pair_transform;
    pcl::PointCloud<PointRGBT>::Ptr cloud_aligned(new pcl::PointCloud<PointRGBT>());

    pcl::PointCloud<PointT> feature_cloud1, feature_cloud2;

    pairAlign(rgb_img1, rgb_img2, depth_img1, depth_img2, cloud_aligned, pair_transform, feature_cloud1, feature_cloud2);

    pcl::PointCloud<PointT> feature_cloud2_aligned;
    pcl::transformPointCloud(feature_cloud2, feature_cloud2_aligned, pair_transform.inverse());

    // Transformed the cloud back to the first frame's reference frame
    *cloud_out += *cloud_aligned;
    viewer->addPointCloud(cloud_out, "cloud_accum");
//    viewer->spin();
//    viewer->removePointCloud("cloud_accum");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color1(boost::make_shared<pcl::PointCloud<PointT>>(feature_cloud1), 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color2(boost::make_shared<pcl::PointCloud<PointT>>(feature_cloud2_aligned), 255, 0, 0);

    // visualize keypoint correspondances in 3D
    viewer->addPointCloud(boost::make_shared<pcl::PointCloud<PointT>>(feature_cloud1), color1,"feature cld 1");
    viewer->addPointCloud(boost::make_shared<pcl::PointCloud<PointT>>(feature_cloud2_aligned), color2,"feature cld 2 aligned");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "feature cld 1");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "feature cld 2 aligned");

    viewer->spin();
}

int main(int argc, const char* argv[])
{
    vector<string> rgb_files = getAllFiles(params.data_root+"rgb.txt");
    vector<string> depth_files = getAllFiles(params.data_root+"depth.txt");

    pcl::PointCloud<PointRGBT>::Ptr cloud_accum(new pcl::PointCloud<PointRGBT>());
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("rgbd point cloud"));

    int mode = params.run_mode;
    switch (mode) {
        case 0:
            runSequence(viewer, rgb_files, depth_files); break;
        case 1:
            int f1_idx, f2_idx;
            // example pair
            string f1("1305031914.865128.png");
            string f2("1305031915.965061.png");
        //    string f1("1305031918.701017.png");
        //    string f2("1305031919.401124.png");
            for (int i=0; i<rgb_files.size(); ++i){
                if (rgb_files[i].find(f1) != string::npos){
                    f1_idx = i;
                    cout<<"f1 index is "<<f1_idx;
                }
                if (rgb_files[i].find(f2) != string::npos){
                    f2_idx = i;
                    cout<<"f2 index is "<<f2_idx;
                }
            }
            runPair(viewer, rgb_files, depth_files, f1_idx, f2_idx);
    }

    return 0;
}