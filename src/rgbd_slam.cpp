//
// Created by yuwei on 5/4/21.
//

#include "rgbd_slam.h"

using namespace std;
using namespace g2o;
using namespace cv;

typedef BlockSolver_6_3 SlamBlockSolver;
typedef LinearSolverCSparse< SlamBlockSolver::PoseMatrixType > SlamLinearSolver;
G2O_USE_TYPE_GROUP(slam3d);


RgbdSLAM::RgbdSLAM(const vector<string>& rgb_files,
                   const vector<string>& depth_files) :
        rgb_files_(rgb_files),
        depth_files_(depth_files),
        cloud_accum_(new pcl::PointCloud<PointRGBT>()),
        viewer_( new pcl::visualization::PCLVisualizer("rgbd slam")),
        kernel_ptr_(RobustKernelFactory::instance()->construct("Cauchy"))
{
    assertm(rgb_files.size() == depth_files.size(), "must have equal number of rgb and depth frames! ");

    // Initialize SIFT detector, g2o solver,..
    detector_ = cv::SiftFeatureDetector::create();

    // Initialize pcl viewer
//    viewer_->createViewPort(0.0, 0.0, 0.5, 1.0, vp1_);
//    viewer_->createViewPort (0.5, 0.0, 1.0, 1.0, vp2_);
//    viewer_->addText("accumulated map: ", 10, 10, "v1 text", vp1_);
//    viewer_->addText("optimized map: ", 10, 10, "v2 text", vp2_);


    // Initialize g2o
    auto linearSolver = g2o::make_unique<SlamLinearSolver>();
    linearSolver->setBlockOrdering(false);

    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<SlamBlockSolver>(move(linearSolver)));
    optimizer_.setAlgorithm(solver);

    // initialize first frame, point cloud
    frame_file_index_ = 0;
    cv::Mat rgb_img, depth_img;
    bool success = readNextRGBDFrame(rgb_img, depth_img);

    if (!success){
        cout<< "No valid data, exiting ..."<<endl;
        return;
    }

    prev_frame_.id = 0;
    prev_frame_.file_id = frame_file_index_;
    frame_file_index_++;

    createPointCloudFromRGBD(rgb_img, depth_img, cloud_accum_);
    downSampleCloud(*cloud_accum_, params.downsample_grid_size);
    statisticalFilter(*cloud_accum_);


    detectFeature(rgb_img, depth_img, detector_, prev_frame_);
    prev_frame_.pose = Eigen::Isometry3f::Identity();

    VertexSE3* v = new VertexSE3();
    v->setId(0);
    v->setEstimate(Eigen::Isometry3d::Identity());
    v->setFixed(true);

    optimizer_.addVertex(v);
    all_frames_.push_back(prev_frame_);
    all_clouds_.push_back(cloud_accum_);

    prev_rgb_img_ = rgb_img;
    prev_cloud_ = cloud_accum_;

    cout<<"SLAM initialized ... "<<endl;
}



void RgbdSLAM::run() {

    cv::Mat rgb_img, depth_img;
    bool success = readNextRGBDFrame(rgb_img, depth_img);
    if (!success) return;

    Frame curr_frame;
    curr_frame.id = all_frames_.size();
    curr_frame.file_id = frame_file_index_;
    frame_file_index_++;

    detectFeature(rgb_img, depth_img, detector_, curr_frame);

    vector<DMatch> matches = findMatches(prev_frame_, curr_frame);
    vector<DMatch> filtered_matches = performRANSAC(prev_frame_, curr_frame, matches);

//    drawMatches(prev_rgb_img_, rgb_img, prev_frame_, curr_frame, matches);
//    drawMatches(prev_rgb_img_, rgb_img, prev_frame_, curr_frame, filtered_matches);

    Eigen::Matrix4f tf = estimatePairTransform(prev_frame_, curr_frame, filtered_matches);

    pcl::PointCloud<PointRGBT>::Ptr curr_cloud (new pcl::PointCloud<PointRGBT>());
//    pcl::PointCloud<PointRGBT>::Ptr curr_cloud_aligned (new pcl::PointCloud<PointRGBT>());
    pcl::PointCloud<PointRGBT>::Ptr cloud_aligned_global (new pcl::PointCloud<PointRGBT>());


    createPointCloudFromRGBD(rgb_img, depth_img, curr_cloud);
    downSampleCloud(*curr_cloud, params.downsample_grid_size);
    statisticalFilter(*curr_cloud);

//    pairAlignCloud(prev_cloud_, curr_cloud, curr_cloud_aligned, tf);

    curr_frame.pose = prev_frame_.pose * tf.inverse();
    pcl::transformPointCloud(*curr_cloud, *cloud_aligned_global, curr_frame.pose.matrix());

    // add as new vertex and add an edge to previous vertex/frame
    VertexSE3* v = new VertexSE3();
    v->setId(curr_frame.id);
    // add some perturbations for testing how well g2o works
    v->setEstimate(curr_frame.pose.cast<double>().rotate(Eigen::AngleAxisd(0.5*((double) rand()/(RAND_MAX)), Eigen::Vector3d(0.3,0.3,0.9))));
//    v->setEstimate(curr_frame.pose.cast<double>().pretranslate(Eigen::Vector3d(0.3*((double) rand()/(RAND_MAX)),
//                                                                               -0.3*((double) rand()/(RAND_MAX)),
//                                                                               0.3*((double) rand()/(RAND_MAX)))));

    optimizer_.addVertex(v);

    EdgeSE3* e = new EdgeSE3();
    e->vertices()[0] = optimizer_.vertex(prev_frame_.id);
    e->vertices()[1] = optimizer_.vertex(curr_frame.id);
    e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
    Eigen::Isometry3d tf_iso; tf_iso = tf.cast<double>();
    e->setMeasurement(tf_iso.inverse());
    optimizer_.addEdge(e);
    cout<<"adding sequential edge between "<<prev_frame_.id << " and "<<curr_frame.id<<endl;


    all_frames_.push_back(curr_frame);
//    all_clouds_.push_back(curr_cloud);

    prev_frame_ = curr_frame;
    prev_rgb_img_ = rgb_img;
    prev_cloud_ = curr_cloud;

    *cloud_accum_ += *cloud_aligned_global;
    downSampleCloud(*cloud_accum_, params.downsample_grid_size);

//    addNeighboringConstraints(curr_frame);

//    viewer_->addPointCloud(cloud_accum_, "cloud_accum");
////    viewer_->addCoordinateSystem(0.2, curr_frame.pose, to_string(curr_frame.id));
//    viewer_->spin();
//    viewer_->removePointCloud("cloud_accum");


}

void RgbdSLAM::addNeighboringConstraints(const Frame& current_frame){
    // skip immediate predecessor since already formed edge
    int neighbor_id = current_frame.id - 2;
    for (int i=0; i<params.num_neighboring_edges; i++){

        if (neighbor_id < 0) return;

        Frame neighbor_frame = all_frames_[neighbor_id];
        vector<DMatch> matches = findMatches(neighbor_frame, current_frame);
        vector<DMatch> filtered_matches = performRANSAC(neighbor_frame, current_frame, matches);

        // check if there are enough matches to form an edge
        if (filtered_matches.size() >= params.min_matches_required){
            Eigen::Matrix4f tf = estimatePairTransform(neighbor_frame, current_frame, filtered_matches);

            EdgeSE3* e = new EdgeSE3();
            e->vertices()[0] = optimizer_.vertex(neighbor_frame.id);
            e->vertices()[1] = optimizer_.vertex(current_frame.id);
            e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
            Eigen::Isometry3d tf_iso; tf_iso = tf.cast<double>();
            e->setMeasurement(tf_iso.inverse());
//            e->setRobustKernel(kernel_ptr_);
            optimizer_.addEdge(e);
            cout<<"adding neighboring edge between "<<neighbor_frame.id << " and "<<current_frame.id<<endl;

        }

        neighbor_id--;
    }
}




bool RgbdSLAM::readNextRGBDFrame(cv::Mat& rgb_img, cv::Mat& depth_img) {

    while (true){
        if (frame_file_index_ > rgb_files_.size()-1){
            cout<<"End of Data Sequence .. "<<endl;
            return false;
        }

        rgb_img = cv::imread(rgb_files_[frame_file_index_],  cv::ImreadModes::IMREAD_UNCHANGED);
        depth_img = cv::imread(depth_files_[frame_file_index_],  cv::ImreadModes::IMREAD_UNCHANGED);

        if (rgb_img.empty() || depth_img.empty()){
            frame_file_index_++;
            continue;
        }
        else{
            return true;
        }
    }
}


void RgbdSLAM::detectFeature(const cv::Mat& rgb_img,
                             const cv::Mat& depth_img,
                             const cv::Ptr<cv::SiftFeatureDetector>& detector,
                             Frame& frame){

    detector->detectAndCompute(rgb_img, cv::noArray(), frame.keypoints, frame.descriptors);

    frame.keypoints3D.clear();
    for (auto& p : frame.keypoints){
        frame.keypoints3D.push_back(pixelTo3DCoord(p.pt, depth_img, true));
    }
}

vector<cv::DMatch> RgbdSLAM::findMatches(const Frame& frame1, const Frame& frame2)
{

    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    vector<vector<cv::DMatch>> all_matches;
    matcher->knnMatch(frame1.descriptors, frame2.descriptors, all_matches, 2);  // for each query descriptor, find only 2 nearest neighbors in search descriptors

    //Only keep a match if best_dist/2n best_dist < ratio_thresh
    const float ratio_thresh = 0.7f;
    vector<cv::DMatch> matches;
    for (int i=0; i<all_matches.size(); ++i){
        if (all_matches[i][0].distance < all_matches[i][1].distance * ratio_thresh){
            cv::DMatch match = all_matches[i][0];
            float query_depth = frame1.keypoints3D[match.queryIdx].z;
            float train_depth = frame2.keypoints3D[match.trainIdx].z;

            if ((params.min_depth <= query_depth && query_depth<= params.max_depth) &&
                ((params.min_depth <= train_depth && train_depth<= params.max_depth))){
                matches.push_back(match);
            }
        }
    }
    return matches;
}

vector<cv::DMatch> RgbdSLAM::performRANSAC(const Frame& frame1,
                                                const Frame& frame2,
                                                const vector<cv::DMatch>& matches)
{
    int max_inlier_count_so_far = 0;
    vector<cv::DMatch> filtered_matches;
    pcl::registration::TransformationEstimation3Point<PointT, PointT> est;

    for (size_t iter=0; iter<params.ransac_max_iter; ++iter){
        vector<cv::DMatch> sampled_matches = sampleSubset(matches, 3);

        // Compute transform using this subset hypothesis
        pcl::PointCloud<PointT>::Ptr cloud_source (new pcl::PointCloud<PointT>);
        pcl::PointCloud<PointT>::Ptr cloud_target (new pcl::PointCloud<PointT>);

        for (auto& match : sampled_matches){
            cloud_source->push_back(frame1.keypoints3D[match.queryIdx]);
            cloud_target->push_back(frame2.keypoints3D[match.trainIdx]);
        }

        Eigen::Matrix4f T;
        est.estimateRigidTransformation(*cloud_source, *cloud_target, T);
        Eigen::Affine3f affine;
        affine = T;
        //evaluate this hypothesis by counting inliers
        vector<cv::DMatch> inliers;

        for (auto& match : matches){
            PointT p_source = frame1.keypoints3D[match.queryIdx];
            PointT p_target = frame2.keypoints3D[match.trainIdx];

            PointT p_aligned = pcl::transformPoint(p_target, affine.inverse());
//            cout<<"reprojection error: "<< pcl::euclideanDistance(p_source, p_aligned) <<endl;

            float temp = (p_source.z - params.min_depth)/(params.max_depth - params.min_depth);
            const float dist_thresh_interp = params.ransac_dist_thresh_low
                                             + temp * (params.ransac_dist_thresh_high - params.ransac_dist_thresh_low);
//            cout<<"dist thresh interp : "<<dist_thresh_interp<<endl;
            if (pcl::euclideanDistance(p_source, p_aligned) < params.ransac_dist_thresh_low){
                inliers.push_back(match);
            }
        }
        //
        if (inliers.size() > max_inlier_count_so_far){
            max_inlier_count_so_far = inliers.size();
            filtered_matches = inliers;
        }
    }

    cout<<"num of filtered matches: "<<filtered_matches.size()<<endl;
    if (filtered_matches.size() < params.min_matches_required){
        cout<<"not enough keypoint correspondances########################! "<<endl;
    }

    return filtered_matches;
};

Eigen::Matrix4f RgbdSLAM::estimatePairTransform(const Frame& frame1, const Frame& frame2, const vector<cv::DMatch> matches){

    pcl::PointCloud<PointT>::Ptr cloud_source (new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr cloud_target (new pcl::PointCloud<PointT>);

    for (auto& match : matches){
        cloud_source->push_back(frame1.keypoints3D[match.queryIdx]);
        cloud_target->push_back(frame2.keypoints3D[match.trainIdx]);
    }

    Eigen::Matrix4f T;
    pcl::registration::TransformationEstimationSVD<PointT, PointT> svd;
    svd.estimateRigidTransformation(*cloud_source, *cloud_target, T);
    return T;
}

void RgbdSLAM::optimizePoseGraph(){
    optimizer_.setVerbose(true);
    optimizer_.initializeOptimization();
    cerr<<"optimizing pose graph ... ..."<<endl;
    optimizer_.optimize(30);
}

void RgbdSLAM::pairAlignCloud(const pcl::PointCloud<PointRGBT>::Ptr& cloud1,
                              const pcl::PointCloud<PointRGBT>::Ptr& cloud2,
                              pcl::PointCloud<PointRGBT>::Ptr& cloud2_aligned,
                              Eigen::Matrix4f & transform,
                              const bool apply_icp){


    if (apply_icp) {
        // icp
        pcl::IterativeClosestPoint<PointRGBT, PointRGBT> icp;
        pcl::PointCloud<PointRGBT> temp;
        icp.setMaximumIterations(params.icp_max_iter);
        icp.setMaxCorrespondenceDistance(params.icp_max_correspondence_dist);
        icp.setTransformationEpsilon(1e-8);

        icp.setInputSource(cloud1);
        icp.setInputTarget(cloud2);
        icp.align(temp, transform);
        transform = icp.getFinalTransformation();

        cout << "converged " << icp.hasConverged() << " score: " <<
                  icp.getFitnessScore() << endl;
    }


    pcl::transformPointCloud(*cloud2, *cloud2_aligned, transform.inverse());

}

void RgbdSLAM::visualizeResultMap() {
//    pcl::PointCloud<PointRGBT>::Ptr result (new pcl::PointCloud<PointRGBT>());
//
//    for (auto& frame : all_frames_){
//        pcl::PointCloud<PointRGBT>::Ptr cld_transformed (new pcl::PointCloud<PointRGBT>());
//
//        VertexSE3* v = dynamic_cast<VertexSE3*>(optimizer_.vertex(frame.id));
//        if (!v) {
//            cout<<"invalid vertex ..."<<endl;
//            continue;
//        }
//
//        pcl::transformPointCloud(*all_clouds_[frame.id], *cld_transformed, v->estimate().matrix());
//        *result += *cld_transformed;
//        downSampleCloud(*result, params.downsample_grid_size);
//
////        viewer_->addCoordinateSystem(0.1, v->estimate().cast<float>(), to_string(frame.id));
//
//    }
    cout<<"visualizing result map..."<<endl;
    viewer_->addPointCloud(cloud_accum_, "accumulated_map");
//    viewer_->addPointCloud(result, "optimized_map", vp2_);

    viewer_->spin();
    viewer_->removePointCloud("accumulated_map");
//    viewer_->removePointCloud("optimized_map");

}


void RgbdSLAM::drawMatches(const cv::Mat& img1, const cv::Mat& img2, const Frame& frame1, const Frame& frame2, const vector<cv::DMatch>& matches){

    cv::Mat result;

    cv::drawMatches(img1, frame1.keypoints, img2, frame2.keypoints, matches, result,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("matches", result);
    cv::waitKey(0);
}
