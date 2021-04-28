//
// Created by yuwei on 4/28/21.
//

#ifndef RGBD_MAPPING_PAIRWISE_ALIGN_H
#define RGBD_MAPPING_PAIRWISE_ALIGN_H

#endif //RGBD_MAPPING_PAIRWISE_ALIGN_H


#include "utils.h"

#include <pcl/common/distances.h>
#include <pcl/registration/transformation_estimation_3point.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <Eigen/Eigen>
#include <numeric>

void findSIFTMatches(const cv::Mat& img1,
                     const cv::Mat& img2,
                     const cv::Mat& depth_img1,
                     const cv::Mat& depth_img2,
                     std::vector<cv::KeyPoint>& keypoints1,
                     std::vector<cv::KeyPoint>& keypoints2,
                     std::vector<cv::DMatch>& matches) {

    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
    cv::Mat descriptors1, descriptors2;

    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    std::vector<std::vector<cv::DMatch>> all_matches;
    matcher->knnMatch(descriptors1, descriptors2, all_matches, 2);  // for each query descriptor, find only 2 nearest neighbors in search descriptors

    //Only keep a match if best_dist/2n best_dist < ratio_thresh
    const float ratio_thresh = 0.7f;
    matches.clear();
    for (int i=0; i<all_matches.size(); ++i){
        if (all_matches[i][0].distance < all_matches[i][1].distance * ratio_thresh){
            cv::DMatch match = all_matches[i][0];
            PointT query_p = pixelTo3DCoord(keypoints1[match.queryIdx].pt, depth_img1, true);
            PointT train_p = pixelTo3DCoord(keypoints2[match.trainIdx].pt, depth_img2, true);

            if ((params.min_depth <= query_p.z && query_p.z <= params.max_depth) &&
                ((params.min_depth <= train_p.z && train_p.z <= params.max_depth))){
                matches.push_back(match);
            }
        }
    }
}


void perform_RANSAC_Alignment(const std::vector<cv::KeyPoint>& keypoints1,
                              const std::vector<cv::KeyPoint>& keypoints2,
                              const std::vector<cv::DMatch>& matches,
                              const cv::Mat& depth_img1,
                              const cv::Mat& depth_img2,
                              std::vector<cv::DMatch>& filtered_matches,
                              Eigen::Matrix4f& Transformation,
                              pcl::PointCloud<PointT>& feature_cloud1,
                              pcl::PointCloud<PointT>& feature_cloud2){


    int max_inlier_count_so_far = 0;

    for (size_t iter=0; iter<params.ransac_max_iter; ++iter){
        std::vector<cv::DMatch> sampled_matches = sampleSubset(matches, 3);

        // Compute transform using this subset hypothesis
        pcl::PointCloud<PointT>::Ptr cloud_source (new pcl::PointCloud<PointT>);
        pcl::PointCloud<PointT>::Ptr cloud_target (new pcl::PointCloud<PointT>);

        for (auto& match : sampled_matches){
            PointT p_source = pixelTo3DCoord(keypoints1[match.queryIdx].pt, depth_img1, true);
            PointT p_target = pixelTo3DCoord(keypoints2[match.trainIdx].pt, depth_img2, true);
            cloud_source->push_back(p_source);
            cloud_target->push_back(p_target);
        }
        Eigen::Matrix4f T;
        pcl::registration::TransformationEstimation3Point<PointT, PointT> svd;
        svd.estimateRigidTransformation(*cloud_source, *cloud_target, T);
        Eigen::Affine3f affine;
        affine = T;

        //evaluate this hypothesis by counting inliers
        std::vector<cv::DMatch> inliers;

        for (auto& match : matches){
            PointT p_source = pixelTo3DCoord(keypoints1[match.queryIdx].pt, depth_img1, true);
            PointT p_target = pixelTo3DCoord(keypoints2[match.trainIdx].pt, depth_img2, true);

            PointT p_aligned = pcl::transformPoint(p_target, affine.inverse());
//            std::cout<<"reprojection error: "<< pcl::euclideanDistance(p_source, p_aligned) <<std::endl;

            float temp = (p_source.z - params.min_depth)/(params.max_depth - params.min_depth);
            const float dist_thresh_interp = params.ransac_min_dist_thresh
                                             + temp * (params.ransac_max_dist_thresh - params.ransac_min_dist_thresh);
//            std::cout<<"dist thresh interp : "<<dist_thresh_interp<<std::endl;
            if (pcl::euclideanDistance(p_source, p_aligned) < dist_thresh_interp){
                inliers.push_back(match);
            }
        }
        //
        if (inliers.size() > max_inlier_count_so_far){
            max_inlier_count_so_far = inliers.size();
            filtered_matches = inliers;
        }
    }
    // Finally, use all inliers to obtain a more accurate transformation
    feature_cloud1.clear();
    feature_cloud2.clear();

    for (auto& match : filtered_matches){
        PointT p_source = pixelTo3DCoord(keypoints1[match.queryIdx].pt, depth_img1, true);
        PointT p_target = pixelTo3DCoord(keypoints2[match.trainIdx].pt, depth_img2, true);
        feature_cloud1.push_back(p_source);
        feature_cloud2.push_back(p_target);
    }

    pcl::registration::TransformationEstimationSVD<PointT, PointT> svd;
    svd.estimateRigidTransformation(feature_cloud1, feature_cloud2, Transformation);

};


bool pairAlign(const cv::Mat& rgb_img1,
               const cv::Mat& rgb_img2,
               const cv::Mat& depth_img1,
               const cv::Mat& depth_img2,
               pcl::PointCloud<PointRGBT>::Ptr& output,
               Eigen::Matrix4f& final_transform,
               pcl::PointCloud<PointT>& feature_cloud1,
               pcl::PointCloud<PointT>& feature_cloud2,
               bool apply_icp = false){

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    std::vector<cv::DMatch> matches;
    findSIFTMatches(rgb_img1, rgb_img2, depth_img1.clone(), depth_img2.clone(), keypoints1, keypoints2, matches);

    Eigen::Matrix4f T;
    std::vector<cv::DMatch> filtered_matches;


    perform_RANSAC_Alignment(keypoints1, keypoints2, matches, depth_img1.clone(), depth_img2.clone(), filtered_matches, T, feature_cloud1, feature_cloud2);

    std::cout<<"num of filtered matches: "<<filtered_matches.size()<<std::endl;

    if (filtered_matches.size() < params.min_matches_required){
        std::cout<<"not enough keypoint correspondances########################! "<<std::endl;
        // Draw 2D matches
    }
    cv::Mat img_matches;
    cv::Mat img_matches_filtered;

    cv::drawMatches(rgb_img1, keypoints1, rgb_img2, keypoints2, matches, img_matches,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("matches", img_matches);
    cv::drawMatches(rgb_img1, keypoints1, rgb_img2, keypoints2, filtered_matches, img_matches_filtered,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("filtered matches", img_matches_filtered);
    cv::waitKey(0);
    pcl::PointCloud<PointRGBT>::Ptr cloud1 (new pcl::PointCloud<PointRGBT>());
    pcl::PointCloud<PointRGBT>::Ptr cloud2 (new pcl::PointCloud<PointRGBT>());
    pcl::PointCloud<PointRGBT>::Ptr cloud_aligned (new pcl::PointCloud<PointRGBT>());

    createPointCloudFromRGBD(cloud1, rgb_img1, depth_img1);
    createPointCloudFromRGBD(cloud2, rgb_img2, depth_img2);

    pcl::PointCloud<PointRGBT>::Ptr cloud1_ds (new pcl::PointCloud<PointRGBT>());
    pcl::PointCloud<PointRGBT>::Ptr cloud2_ds (new pcl::PointCloud<PointRGBT>());

    downSampleCloud(*cloud1, *cloud1_ds, params.downsample_grid_size);
    downSampleCloud(*cloud2, *cloud2_ds, params.downsample_grid_size);

    statistical_filter(*cloud1_ds);
    statistical_filter(*cloud2_ds);

    if (apply_icp) {
        // icp
        pcl::IterativeClosestPoint<PointRGBT, PointRGBT> icp;
        icp.setMaximumIterations(params.icp_max_iter);
        icp.setMaxCorrespondenceDistance(params.icp_max_correspondence_dist);
        icp.setTransformationEpsilon(1e-8);

        icp.setInputSource(cloud1_ds);
        icp.setInputTarget(cloud2_ds);
        icp.align(*cloud_aligned, T);
        final_transform = icp.getFinalTransformation();

        std::cout << "converged " << icp.hasConverged() << " score: " <<
                  icp.getFitnessScore() << std::endl;
    }
    else{
        final_transform = T;
    }

    pcl::transformPointCloud(*cloud2_ds, *output, final_transform.inverse());

    return true;
}
