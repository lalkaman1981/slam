#include "bench.hpp"

// N 3D-3D pairs (30% outliers) and N 3D-2D observations (10% outliers);
// measures RANSAC on the pairs and then pose refinement on the observations
long long run(const std::string& path) {
    const auto input = read_pose_input(path);

#ifdef BENCH_OPENCV
    std::vector<cv::Point3d> from, to, object;
    std::vector<cv::Point2d> image;
    for (const auto& p : input.pairs) {
        from.push_back(p.a);
        to.push_back(p.b);
    }
    for (const auto& o : input.observations) {
        object.push_back(o.pointWorld);
        image.push_back(o.pixel);
    }
    const cv::Mat K(input.K);
    cv::Mat affine, inliers, rvec, tvec;

    auto start_time = get_current_time_fenced();
    cv::estimateAffine3D(from, to, affine, inliers, RANSAC_THRESHOLD, RANSAC_CONFIDENCE);
    rigid_from_affine(affine, rvec, tvec);
    cv::solvePnPRefineLM(object, image, K, cv::noArray(), rvec, tvec, refine_criteria());
    auto finish_time = get_current_time_fenced();
    auto total_time = finish_time - start_time;

    sink = sink + cv::countNonZero(inliers) + micro(tvec.at<double>(0));
#else
    auto start_time = get_current_time_fenced();
    const auto ransac = estimatePoseRansac(input.pairs);
    const auto refined = refinePose(input.observations, ransac ? ransac->T_ba : cv::Affine3d::Identity(), input.K);
    auto finish_time = get_current_time_fenced();
    auto total_time = finish_time - start_time;

    sink = sink + (ransac ? static_cast<long long>(ransac->numInliers) : -1) +
           (refined ? micro(refined->T_cw.translation()[0]) : -1);
#endif
    return to_ns(total_time);
}

int main(int argc, char** argv) {
    const command_line_options_t options(argc, argv);
    std::cout << run(options.get_input_file()) << std::endl;
}
