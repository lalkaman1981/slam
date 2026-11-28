#include "bench.hpp"
#include "integration_adapters.hpp"
#include "linalg.hpp"
#include "pose_graph.hpp"
#include "se3.hpp"

#ifdef BENCH_OPENCV
#include <opencv2/features2d.hpp>
#endif

// mapping step of both chains: one Gauss-Newton step of the pose graph over the sequence
// (odometry edges, first pose fixed). The linearization is ours in both chains; the variants
// differ in the linear solver only, because OpenCV has no SLAM mapping.
slam::GraphSystem pose_graph_system(const std::vector<cv::Affine3d>& T_cw) {
    slam::PoseGraph graph;
    for (size_t k = 0; k < T_cw.size(); ++k)
        slam::addGraphNode(graph, static_cast<int>(k), slam::toMappingPose(T_cw[k]), k == 0);
    for (size_t k = 1; k < T_cw.size(); ++k) {
        const slam::Pose relative = slam::composePose(slam::toMappingPose(T_cw[k - 1]),
                                                      slam::inversePose(slam::toMappingPose(T_cw[k])));
        slam::addOdometryEdge(graph, static_cast<int>(k - 1), static_cast<int>(k), relative);
    }
    return slam::linearizePoseGraph(graph, slam::ExecutionMode::Serial);
}

// a synthetic RGB-D sequence; measures the whole chain: features, matching and pose for every
// frame against the previous one, then the mapping solve over all poses
long long run(const std::string& path) {
    const auto sequence = read_sequence(path);
    const auto& K = sequence.K;
    const size_t frames = sequence.gray.size();
    std::vector<cv::Affine3d> T_cw(frames, cv::Affine3d::Identity());

#ifdef BENCH_OPENCV
    const auto orb = cv::ORB::create(1000);
    const cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::KeyPoint> previous_keypoints, keypoints;
    cv::Mat previous_descriptors, descriptors, affine, inliers, rvec, tvec, delta;
    std::vector<cv::DMatch> matches;
    std::vector<cv::Point3d> from, to, object;
    std::vector<cv::Point2d> image, inlier_image;
    size_t tracked = 0;

    auto start_time = get_current_time_fenced();
    orb->detectAndCompute(sequence.gray[0], cv::noArray(), previous_keypoints, previous_descriptors);
    for (size_t k = 1; k < frames; ++k) {
        orb->detectAndCompute(sequence.gray[k], cv::noArray(), keypoints, descriptors);
        matcher.match(previous_descriptors, descriptors, matches);
        // 3D points from depth with person 2's depth rules, as in build3DCorrespondences
        const ::Frame previous_depth{{}, sequence.depth[k - 1]}, current_depth{{}, sequence.depth[k]};
        from.clear(), to.clear(), image.clear();
        for (const auto& m : matches) {
            const cv::Point2d pa = previous_keypoints[m.queryIdx].pt, pb = keypoints[m.trainIdx].pt;
            const auto za = readValidDepth(previous_depth, cv::Point(cvRound(pa.x), cvRound(pa.y)));
            const auto zb = readValidDepth(current_depth, cv::Point(cvRound(pb.x), cvRound(pb.y)));
            if (!za || !zb) continue;
            from.push_back(backProject(cv::Point2d(cvRound(pa.x), cvRound(pa.y)), *za, K));
            to.push_back(backProject(cv::Point2d(cvRound(pb.x), cvRound(pb.y)), *zb, K));
            image.push_back(pb);
        }
        T_cw[k] = T_cw[k - 1];
        if (from.size() >= MIN_INLIERS && cv::estimateAffine3D(from, to, affine, inliers, RANSAC_THRESHOLD, RANSAC_CONFIDENCE)) {
            rigid_from_affine(affine, rvec, tvec);
            object.clear(), inlier_image.clear();
            for (int i = 0; i < inliers.rows * inliers.cols; ++i)
                if (inliers.data[i]) object.push_back(from[i]), inlier_image.push_back(image[i]);
            if (object.size() >= MIN_INLIERS) {
                cv::solvePnPRefineLM(object, inlier_image, cv::Mat(K), cv::noArray(), rvec, tvec, refine_criteria());
                T_cw[k] = cv::Affine3d(cv::Vec3d(rvec), cv::Vec3d(tvec)) * T_cw[k - 1];
                ++tracked;
            }
        }
        std::swap(previous_keypoints, keypoints);
        std::swap(previous_descriptors, descriptors);
    }
    const auto system = pose_graph_system(T_cw);
    const size_t n = system.b.size();
    const cv::Mat H(static_cast<int>(n), static_cast<int>(n), CV_64F, const_cast<double*>(system.H.row(0).data()));
    cv::Mat rhs = -cv::Mat(system.b, true);
    const bool solved = cv::solve(H, rhs, delta, cv::DECOMP_CHOLESKY);
    auto finish_time = get_current_time_fenced();
    auto total_time = finish_time - start_time;

    sink = sink + static_cast<long long>(tracked) + (solved ? micro(delta.at<double>(0)) : -1);
#else
    std::vector<cv::Point3d> map_points;
    orb_matches_t map_matches;
    std::vector<double> delta;
    size_t tracked = 0;

    auto start_time = get_current_time_fenced();
    ::Frame previous{extractFeatures(sequence.gray[0], 1000), sequence.depth[0]};
    for (size_t k = 1; k < frames; ++k) {
        ::Frame current{extractFeatures(sequence.gray[k], 1000), sequence.depth[k]};
        const auto matches = match_frames(previous.features, current.features);
        // local map for refinement: the previous frame's matched points in world coordinates
        map_points.clear(), map_matches.clear();
        const cv::Affine3d T_wc = T_cw[k - 1].inv();
        for (const auto& pair : build3DCorrespondences(previous, current, matches, K)) {
            map_matches[pair.idxB] = map_points.size();
            map_points.push_back(T_wc * pair.a);
        }
        const auto track = trackFrame(current, previous, T_cw[k - 1], matches, map_points, map_matches, K);
        T_cw[k] = track.status == TrackingStatus::OK ? track.T_cw : T_cw[k - 1];
        tracked += track.status == TrackingStatus::OK;
        previous = std::move(current);
    }
    const auto system = pose_graph_system(T_cw);
    std::vector<double> rhs(system.b.size());
    for (size_t i = 0; i < rhs.size(); ++i) rhs[i] = -system.b[i];
    const bool solved = slam::solveCholesky(system.H, rhs, delta, slam::ExecutionMode::Serial);
    auto finish_time = get_current_time_fenced();
    auto total_time = finish_time - start_time;

    sink = sink + static_cast<long long>(tracked) + (solved ? micro(delta[0]) : -1);
#endif
    return to_ns(total_time);
}

int main(int argc, char** argv) {
    const command_line_options_t options(argc, argv);
    std::cout << run(options.get_input_file()) << std::endl;
}
