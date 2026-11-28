#include "bench.hpp"

#ifdef BENCH_OPENCV
#include <opencv2/features2d.hpp>
#endif

// two frames of a synthetic sequence; measures feature extraction on both frames and matching between them
long long run(const std::string& input) {
    const auto sequence = read_sequence(input);
    const cv::Mat& first = sequence.gray.at(0);
    const cv::Mat& second = sequence.gray.at(1);

#ifdef BENCH_OPENCV
    const auto orb = cv::ORB::create(1000);
    const cv::BFMatcher matcher(cv::NORM_HAMMING, true);
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    std::vector<cv::DMatch> matches;

    auto start_time = get_current_time_fenced();
    orb->detectAndCompute(first, cv::noArray(), keypoints1, descriptors1);
    orb->detectAndCompute(second, cv::noArray(), keypoints2, descriptors2);
    matcher.match(descriptors1, descriptors2, matches);
    auto finish_time = get_current_time_fenced();
    auto total_time = finish_time - start_time;

    sink = sink + static_cast<long long>(keypoints1.size() + keypoints2.size() + matches.size());
#else
    auto start_time = get_current_time_fenced();
    const auto features1 = extractFeatures(first, 1000);
    const auto features2 = extractFeatures(second, 1000);
    const auto matches = match_frames(features1, features2);
    auto finish_time = get_current_time_fenced();
    auto total_time = finish_time - start_time;

    sink = sink + static_cast<long long>(features1.size() + features2.size() + matches.size());
#endif
    return to_ns(total_time);
}

int main(int argc, char** argv) {
    const command_line_options_t options(argc, argv);
    std::cout << run(options.get_input_file()) << std::endl;
}
