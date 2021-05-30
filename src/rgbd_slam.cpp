
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
        kernel_ptr_(RobustKernelFactory::instance()->construct("Huber"))
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
    prev_frame_.is_keyframe = true;
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
    keyframes_.push_back(prev_frame_);

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
    pcl::PointCloud<PointRGBT>::Ptr cloud_aligned_global (new pcl::PointCloud<PointRGBT>());

//    createPointCloudFromRGBD(rgb_img, depth_img, curr_cloud);
//    downSampleCloud(*curr_cloud, params.downsample_grid_size);
//    statisticalFilter(*curr_cloud);

//    pairAlignCloud(prev_cloud_, curr_cloud, curr_cloud_aligned, tf);

    curr_frame.pose = prev_frame_.pose * tf.inverse();

    // Add to pose graph
    VertexSE3* v = new VertexSE3();
    v->setId(curr_frame.id);
    Eigen::Isometry3d est = curr_frame.pose.cast<double>();
//    est.rotate(Eigen::AngleAxisd(0.4*((double) rand()/(RAND_MAX)), Eigen::Vector3d(0.3,0.3,0.9)));
//    est.pretranslate(Eigen::Vector3d(0.3*((double) rand()/(RAND_MAX)), -0.3*((double) rand()/(RAND_MAX)), 0.3*((double) rand()/(RAND_MAX))));
    v->setEstimate(est);
    optimizer_.addVertex(v);

    EdgeSE3* e = new EdgeSE3();
    e->vertices()[0] = optimizer_.vertex(prev_frame_.id);
    e->vertices()[1] = optimizer_.vertex(curr_frame.id);
    e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
    Eigen::Isometry3d tf_iso; tf_iso = tf.cast<double>();
    e->setMeasurement(tf_iso.inverse());
//    e->setRobustKernel(kernel_ptr_);
    optimizer_.addEdge(e);
    cout<<"adding sequential edge between "<<prev_frame_.id << " and "<<curr_frame.id<<endl;

    addNeighboringConstraints(curr_frame);

    if (isNewKeyframe(keyframes_.back(), curr_frame)){
        // set current frame to keyframe
        curr_frame.is_keyframe = true;
        // add as new vertex and add an edge to previous vertex/frame

        int loop_closure_count = checkLoopClosure(curr_frame);
        if (loop_closure_count > 0) cout << "Detected "<<loop_closure_count<<" loop closures !!"<<endl;

        keyframes_.push_back(curr_frame);
        cout << "Total num of keyframes so far "<<keyframes_.size()<<endl;
    }

    all_frames_.push_back(curr_frame);

    prev_frame_ = curr_frame;
    prev_rgb_img_ = rgb_img;
    prev_cloud_ = curr_cloud;

//    pcl::transformPointCloud(*curr_cloud, *cloud_aligned_global, curr_frame.pose.matrix());
//    *cloud_accum_ += *cloud_aligned_global;

//    if (curr_frame.id % 10 ==0){
//        // downsample cloud only every 10 runs for efficiency
//        downSampleCloud(*cloud_accum_, params.downsample_grid_size);
//        statisticalFilter(*cloud_accum_);
//    }

//    viewer_->addPointCloud(cloud_accum_, "cloud_accum");
////    viewer_->addCoordinateSystem(0.2, curr_frame.pose, to_string(curr_frame.id));
//    viewer_->spin();
//    viewer_->removePointCloud("cloud_accum");

}



bool RgbdSLAM::isNewKeyframe(const Frame& prev_keyframe, const Frame& curr_frame){
    vector<DMatch> matches = findMatches(prev_keyframe, curr_frame);
    vector<DMatch> filtered_matches = performRANSAC(prev_keyframe, curr_frame, matches);
    return filtered_matches.size() < params.keyframe_thresh ? true : false;
}

void RgbdSLAM::addNeighboringConstraints(const Frame& current_frame){
    if (current_frame.id<2) return;
    // skip immediate predecessor since already formed edge
    int idx =current_frame.id-2;
    for (int i=0; i<params.num_neighboring_edges; i++){

        if (idx < 0) return;

        Frame neighbor_frame = all_frames_[idx];
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
//            cout<<"adding neighboring edge between "<<neighbor_frame.id << " and "<<current_frame.id<<endl;

            if (checkDistancePreserved(neighbor_frame, current_frame, filtered_matches))  cout <<"Yes"<< endl;
            else cout <<"No"<< endl;

//            const Mat img1 = cv::imread(rgb_files_[neighbor_frame.file_id],  cv::ImreadModes::IMREAD_UNCHANGED);
//            const Mat img2 = cv::imread(rgb_files_[current_frame.file_id],  cv::ImreadModes::IMREAD_UNCHANGED);
//            drawMatches(img1, img2, neighbor_frame, current_frame, filtered_matches);
        }
        idx--;
    }
}

int RgbdSLAM::checkLoopClosure(const Frame& curr_keyframe){

    // sample a few keyframes from all previous keyframes (except the most recent predecessors)
    if (keyframes_.size()<= params.num_neighboring_edges) return 0;
    vector<int> sampled_indices = sampleSubset(keyframes_.size()-params.num_neighboring_edges,
                                               params.num_loop_closure_frames);
    if (sampled_indices.empty()) return 0;
    int loop_closure_count = 0;


    for(const int& idx : sampled_indices){
        Frame keyframe = keyframes_[idx];
        vector<DMatch> matches = findMatches(keyframe, curr_keyframe);
        vector<DMatch> filtered_matches = performRANSAC(keyframe, curr_keyframe, matches, true);

        if (filtered_matches.size() >= params.min_matches_required){
            bool dist_preserved = checkDistancePreserved(keyframe, curr_keyframe, filtered_matches);
            cout <<"distanced preserved ? "<< dist_preserved<< endl;
            // Draw matches for debugg
            const Mat img1 = cv::imread(rgb_files_[keyframe.file_id],  cv::ImreadModes::IMREAD_UNCHANGED);
            const Mat img2 = cv::imread(rgb_files_[curr_keyframe.file_id],  cv::ImreadModes::IMREAD_UNCHANGED);
            drawMatches(img1, img2, keyframe, curr_keyframe, filtered_matches);

            if (!dist_preserved)  continue;

            Eigen::Matrix4f tf = estimatePairTransform(keyframe, curr_keyframe, filtered_matches);

            EdgeSE3* e = new EdgeSE3();
            e->vertices()[0] = optimizer_.vertex(keyframe.id);
            e->vertices()[1] = optimizer_.vertex(curr_keyframe.id);
            e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
            Eigen::Isometry3d tf_iso; tf_iso = tf.cast<double>();
            e->setMeasurement(tf_iso.inverse());
//            e->setRobustKernel(kernel_ptr_);

            optimizer_.addEdge(e);
            cout<<"Loop Closing !!! ******************************************************************** "<<endl;
            cout<<"adding loop closure edge between "<<keyframe.id << " and "<<curr_keyframe.id<<endl;
            loop_closure_count++;

            //sample in neighborhood of the two
            searchMoreLoopClosures(keyframe.id, curr_keyframe.id);
        }
    }
    return loop_closure_count;
}


int RgbdSLAM::searchMoreLoopClosures(const int loc1, const int loc2){
    // sample in the neighborhood of the two locations
    for (int i=0; i<10; i++){
        int idx1 = max(loc1-5, 0) + (rand() % 10);
        idx1 = min(idx1, int(all_frames_.size()-1));

        int idx2 = max(loc2-10, 0) + (rand() % 10);
        idx2 = min(idx2, int(all_frames_.size()-1));

        Frame frame1 = all_frames_[idx1];
        Frame frame2 = all_frames_[idx2];
        vector<DMatch> matches = findMatches(frame1, frame2);
        vector<DMatch> filtered_matches = performRANSAC(frame1, frame2, matches, true);

        if (filtered_matches.size() >= params.min_matches_required) {
            bool dist_preserved = checkDistancePreserved(frame1, frame2, filtered_matches);
            cout << "distanced preserved ? " << dist_preserved << endl;

            if (!dist_preserved){
                const Mat img1 = cv::imread(rgb_files_[frame1.file_id], cv::ImreadModes::IMREAD_UNCHANGED);
                const Mat img2 = cv::imread(rgb_files_[frame2.file_id], cv::ImreadModes::IMREAD_UNCHANGED);
                drawMatches(img1, img2, frame1, frame2, filtered_matches);
                continue;
            }

            Eigen::Matrix4f tf = estimatePairTransform(frame1, frame2, filtered_matches);

            EdgeSE3 *e = new EdgeSE3();
            e->vertices()[0] = optimizer_.vertex(frame1.id);
            e->vertices()[1] = optimizer_.vertex(frame2.id);
            e->setInformation(Eigen::Matrix<double, 6, 6>::Identity());
            Eigen::Isometry3d tf_iso;
            tf_iso = tf.cast<double>();
            e->setMeasurement(tf_iso.inverse());
//            e->setRobustKernel(kernel_ptr_);

            optimizer_.addEdge(e);
            cout << "More loop closing *******" << "between " << frame1.id << " and " << frame2.id << endl;
        }
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
                                           const vector<cv::DMatch>& matches,
                                           const bool loop_closure)
{
    int max_inlier_count_so_far = 0;
    vector<cv::DMatch> filtered_matches;
    pcl::registration::TransformationEstimation3Point<PointT, PointT> est;

    for (size_t iter=0; iter<params.ransac_max_iter; ++iter){
        vector<int> sampled_indices = sampleSubset(matches.size(), 3);
        if (sampled_indices.size() < 3) continue;
        // Compute transform using this subset hypothesis
        pcl::PointCloud<PointT>::Ptr cloud_source (new pcl::PointCloud<PointT>);
        pcl::PointCloud<PointT>::Ptr cloud_target (new pcl::PointCloud<PointT>);


        for (const int& idx : sampled_indices){
            DMatch match = matches[idx];
            cloud_source->push_back(frame1.keypoints3D[match.queryIdx]);
            cloud_target->push_back(frame2.keypoints3D[match.trainIdx]);
        }

        Eigen::Matrix4f T;
        est.estimateRigidTransformation(*cloud_source, *cloud_target, T);
        Eigen::Affine3f affine;
        affine = T;
        //evaluate this hypothesis by counting inliers
        vector<cv::DMatch> inliers;

        for (const auto& match : matches){
            PointT p_source = frame1.keypoints3D[match.queryIdx];
            PointT p_target = frame2.keypoints3D[match.trainIdx];

            PointT p_aligned = pcl::transformPoint(p_target, affine.inverse());
//            cout<<"reprojection error: "<< pcl::euclideanDistance(p_source, p_aligned) <<endl;

            float temp = (p_source.z - params.min_depth)/(params.max_depth - params.min_depth);
            float dist_thresh_interp = params.ransac_dist_thresh_low
                                             + temp * (params.ransac_dist_thresh_high - params.ransac_dist_thresh_low);
//            cout<<"dist thresh interp : "<<dist_thresh_interp<<endl;
            if (loop_closure) {
                dist_thresh_interp = params.ransac_dist_thresh_low
                                           + temp * (params.lc_dist_thresh - params.ransac_dist_thresh_low);
            }

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

//    cout<<"num of filtered matches: "<<filtered_matches.size()<<endl;
//    if (filtered_matches.size() < params.min_matches_required){
//        cout<<"not enough keypoint correspondances########################! "<<endl;
//    }

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

    Eigen::Affine3f affine; affine = T;

    float error_sum = 0;
    for (const auto& match : matches){
        PointT p_source = frame1.keypoints3D[match.queryIdx];
        PointT p_target = frame2.keypoints3D[match.trainIdx];

        PointT p_aligned = pcl::transformPoint(p_target, affine.inverse());
//            cout<<"reprojection error: "<< pcl::euclideanDistance(p_source, p_aligned) <<endl;

        error_sum += pcl::euclideanDistance(p_source, p_aligned);
    }
//    cout << "mean transform error: " << error_sum/matches.size() <<endl;
    return T;
}


float getMeanDistToCentroid(const vector<PointT>& points){
    Vector3F sum3d(0, 0, 0);
    for (const auto &p : points) {
        sum3d += Vector3F(p.x, p.y, p.z);
    }
    Vector3F centroid = sum3d / points.size();

    float sum = 0;
    for (const auto &p : points) {
        sum += (Vector3F(p.x, p.y, p.z) - centroid).norm();
    }
    float mean_dist = sum/points.size();

    return mean_dist;
}

bool RgbdSLAM::checkDistancePreserved(const Frame& frame1, const Frame& frame2, vector<DMatch> matches) {
    // randomly select a few pairs and check if distance is preserved
//    int bad_count = 0;
//    for (int iter=0; iter < 15; iter++){
//
//        vector<int> subset = sampleSubset(matches.size(), 2);
//        DMatch match1 = matches[subset[0]];
//        DMatch match2 = matches[subset[1]];
//        float dist1 = pcl::euclideanDistance(frame1.keypoints3D[match1.queryIdx],
//                                             frame1.keypoints3D[match2.queryIdx]);
//        float dist2 = pcl::euclideanDistance(frame2.keypoints3D[match1.trainIdx],
//                                             frame2.keypoints3D[match2.trainIdx]);
//
//
//        if (min(dist1, dist2)/max(dist1, dist2) <0.60){
//
//            if (dist1< 0.1 && dist2< 0.1) continue;
//
////            cout<<"dist1: "<<dist1<<" dist2: "<<dist2<<endl;
////            cout<<"dist ratio"<<min(dist1, dist2)/max(dist1, dist2)<<endl;
//            bad_count++;
//        }
//
//        if (bad_count > 2) return false;
//    }
//    return true;
    pcl::PointCloud<PointT>::Ptr pts1_cld (new pcl::PointCloud<PointT>());
    pcl::PointCloud<PointT>::Ptr pts2_cld (new pcl::PointCloud<PointT>());

    for(const auto& match : matches){
        pts1_cld->push_back(frame1.keypoints3D[match.queryIdx]);
        pts2_cld->push_back(frame2.keypoints3D[match.trainIdx]);
    }
    statisticalFilter(*pts1_cld);
    statisticalFilter(*pts2_cld);

    vector<PointT> pts1(pts1_cld->points.begin(), pts1_cld->points.end());
    vector<PointT> pts2(pts2_cld->points.begin(), pts2_cld->points.end());

    float mean_dist1 = getMeanDistToCentroid(pts1);
    float mean_dist2 = getMeanDistToCentroid(pts2);
//    cout<<"std1: "<<std1<<"   std2: "<<std2<<endl;
    if (min(mean_dist1, mean_dist2)/max(mean_dist1, mean_dist2) < 0.8){
        cout<<"ratio: "<<min(mean_dist1, mean_dist2)/max(mean_dist1, mean_dist2);
    }
    return min(mean_dist1, mean_dist2) / max(mean_dist1, mean_dist2) > 0.8;
}


void RgbdSLAM::optimizePoseGraph(){
    optimizer_.setVerbose(true);
    optimizer_.initializeOptimization();
    cerr<<"optimizing pose graph ... ..."<<endl;
    optimizer_.optimize(50);

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



void RgbdSLAM::visualizeWholeMap(){
//    pcl::PointCloud<PointRGBT>::Ptr result (new pcl::PointCloud<PointRGBT>());
//    for (auto& kf : keyframes_){
//        int idx = kf.id + 1;
//
//        VertexSE3* v = dynamic_cast<VertexSE3*>(optimizer_.vertex(kf.id));
//        if (!v) {
//            cout<<"invalid vertex ..."<<endl;
//            continue;
//        }
//        Eigen::Matrix4f kf_pose_new = v->estimate().matrix().cast<float>();
//
//        while (!all_frames_[idx].is_keyframe and idx < all_frames_.size()){
//            pcl::PointCloud<PointRGBT>::Ptr cld (new pcl::PointCloud<PointRGBT>());
//            pcl::PointCloud<PointRGBT>::Ptr cld_transformed (new pcl::PointCloud<PointRGBT>());
//
//            Eigen::Matrix4f T_new = kf_pose_new * (kf.pose.inverse() * all_frames_[idx].pose).matrix();
//            createPointCloudFromFile(all_frames_[idx].file_id, cld);
//            pcl::transformPointCloud(*cld, *cld_transformed, T_new);
//            *result += *cld_transformed;
//            idx++;
//        }
//        downSampleCloud(*result, params.downsample_grid_size);
//        viewer_->addCoordinateSystem(0.1, v->estimate().cast<float>(), to_string(kf.id));
//
//        viewer_->addPointCloud(result, "whole_map");
//        viewer_->spin();
//        viewer_->removePointCloud("whole_map");
//    }

    pcl::PointCloud<PointRGBT>::Ptr result (new pcl::PointCloud<PointRGBT>());

    for (int id=0; id<all_frames_.size(); id+= 5){
        Frame f = all_frames_[id];
        pcl::PointCloud<PointRGBT>::Ptr cld (new pcl::PointCloud<PointRGBT>());
        pcl::PointCloud<PointRGBT>::Ptr cld_transformed (new pcl::PointCloud<PointRGBT>());

        VertexSE3* v = dynamic_cast<VertexSE3*>(optimizer_.vertex(f.id));
        if (!v) {
            cout<<"invalid vertex ..."<<endl;
            continue;
        }

        createPointCloudFromFile(f.file_id, cld);
        pcl::transformPointCloud(*cld, *cld_transformed, v->estimate().matrix());
        *result += *cld_transformed;

//        if (count % 10 == 0) downSampleCloud(*result, params.downsample_grid_size);
        viewer_->addCoordinateSystem(0.1, v->estimate().cast<float>(), to_string(f.id));
    }

    downSampleCloud(*result, params.downsample_grid_size);
    cout<<"visualizing full map..."<<endl;
    viewer_->addPointCloud(result, "keyframe_map");
//    viewer_->addPointCloud(result, "optimized_map", vp2_);

    viewer_->spin();
    viewer_->removePointCloud("full_map");
    cout<<"visualizing result map..."<<endl;
//    viewer_->addPointCloud(cloud_accum_, "accumulated_map");
////    viewer_->addPointCloud(result, "optimized_map", vp2_);
//
//    viewer_->spin();
//    viewer_->removePointCloud("accumulated_map");
}

void RgbdSLAM::visualizeKeyframeMap() {
    pcl::PointCloud<PointRGBT>::Ptr result (new pcl::PointCloud<PointRGBT>());

    int count = 0;
    for (auto& kf : keyframes_){
        pcl::PointCloud<PointRGBT>::Ptr cld (new pcl::PointCloud<PointRGBT>());
        pcl::PointCloud<PointRGBT>::Ptr cld_transformed (new pcl::PointCloud<PointRGBT>());

        VertexSE3* v = dynamic_cast<VertexSE3*>(optimizer_.vertex(kf.id));
        if (!v) {
            cout<<"invalid vertex ..."<<endl;
            continue;
        }

        createPointCloudFromFile(kf.file_id, cld);
        pcl::transformPointCloud(*cld, *cld_transformed, v->estimate().matrix());
        *result += *cld_transformed;

//        if (count % 10 == 0) downSampleCloud(*result, params.downsample_grid_size);
        count++;
        viewer_->addCoordinateSystem(0.1, v->estimate().cast<float>(), to_string(kf.id));
    }

    downSampleCloud(*result, params.downsample_grid_size);
    cout<<"visualizing keyframe map..."<<endl;
    viewer_->addPointCloud(result, "keyframe_map");
//    viewer_->addPointCloud(result, "optimized_map", vp2_);

    viewer_->spin();
    viewer_->removePointCloud("keyframe_map");

}

void RgbdSLAM::createPointCloudFromFile(const int& file_id, pcl::PointCloud<PointRGBT>::Ptr& output_cloud){
    const Mat rgb_img = cv::imread(rgb_files_[file_id],  cv::ImreadModes::IMREAD_UNCHANGED);
    const Mat depth_img = cv::imread(depth_files_[file_id],  cv::ImreadModes::IMREAD_UNCHANGED);
    createPointCloudFromRGBD(rgb_img, depth_img, output_cloud);
}


void RgbdSLAM::drawMatches(const cv::Mat& img1, const cv::Mat& img2, const Frame& frame1, const Frame& frame2, const vector<cv::DMatch>& matches){

    cv::Mat result;

    cv::drawMatches(img1, frame1.keypoints, img2, frame2.keypoints, matches, result,
                    cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("matches", result);
    cv::waitKey(0);
}