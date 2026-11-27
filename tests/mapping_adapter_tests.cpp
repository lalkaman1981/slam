#include "integration_adapters.hpp"
#include "teammate_interfaces.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <numbers>
#include <random>
#include <stdexcept>
#include <string>

namespace {

void check(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

void close(double actual, double expected, double tolerance, const std::string &message) {
    check(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

void checkPose(const slam::Pose &actual, const slam::Pose &expected, double tolerance) {
    for (std::size_t row = 0; row < 3; ++row) {
        close(actual.t[row], expected.t[row], tolerance, "pose translation");
        for (std::size_t col = 0; col < 3; ++col)
            close(actual.R(row, col), expected.R(row, col), tolerance, "pose rotation");
    }
}

void featureConversion() {
    slam::CameraIntrinsics camera;
    ::Frame input;
    input.depth = cv::Mat(camera.height, camera.width, CV_16UC1, cv::Scalar(0));
    const brief_desc_t bits{0x0123456789abcdefLL, -1, std::numeric_limits<long long>::min(), 42};
    input.features.push_back({{12.0f, {123, 77}, 3}, bits, static_cast<float>(std::numbers::pi / 3.0)});
    input.features.push_back({{1.0f, {42, 42}, 0}, bits, 0.0f});
    const auto [u, v] = ::scale_coords({123, 77}, std::pow(1.3, 3));
    input.depth.at<std::uint16_t>(v, u) = 12500;
    const slam::Frame frame = slam::toMappingFrame(input, 7, 1.25, camera);
    check(frame.id == 7 && frame.timestamp == 1.25, "frame identity");
    // keypoints are stored as float, so compare with the float-rounded expectation
    close(frame.keypoints[0].u, static_cast<float>(123 * std::pow(1.3, 3)), 1e-5, "level-0 x");
    close(frame.keypoints[0].v, static_cast<float>(77 * std::pow(1.3, 3)), 1e-5, "level-0 y");
    close(frame.keypoints[0].angle, 60.0, 1e-5, "radians to degrees");
    close(frame.depths[0], 2.5, 1e-6, "depth conversion");
    check(frame.depths[1] == 0.0f, "invalid depth retained as zero");
    check(slam::toTrackingDescriptor(frame.descriptors[0]) == bits, "all 256 descriptor bits preserved");
    const auto reconstructed = slam::toTrackingFeatures(frame);
    check(reconstructed[0].candidate.coords == input.features[0].candidate.coords,
          "level-local pixel roundtrip");
    check(reconstructed[0].candidate.level == 3, "octave roundtrip");
    close(reconstructed[0].orientation, input.features[0].orientation, 1e-6, "angle roundtrip");
    ::Frame invalid = input;
    invalid.depth = cv::Mat(camera.height, camera.width, CV_32FC1);
    bool rejected = false;
    try {
        (void)slam::toMappingFrame(invalid, 0, 0.0, camera);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "incorrect depth type rejected");
}

void poseConversion() {
    for (double angle : {0.0, 1e-10, 0.2, std::numbers::pi - 1e-5}) {
        slam::Vec6 twist;
        twist[0] = angle;
        twist[3] = 0.15;
        twist[4] = -0.2;
        twist[5] = 0.3;
        const slam::Pose pose = slam::expSE3(twist);
        const cv::Affine3d upstream = ::expSE3({0.15, -0.2, 0.3, angle, 0.0, 0.0});
        checkPose(pose, slam::toMappingPose(upstream), 1e-12);
        checkPose(slam::toMappingPose(slam::toTrackingPose(pose)), pose, 1e-12);
        checkPose(slam::composePose(pose, slam::inversePose(pose)), slam::Pose{}, 1e-10);
        const slam::Vec6 recovered = slam::logSE3(pose);
        for (std::size_t i = 0; i < 6; ++i)
            close(recovered[i], twist[i], 1e-9, "twist order and SE3 roundtrip");
    }
}

void loopVerification() {
    slam::CameraIntrinsics camera;
    slam::Frame candidate, current;
    candidate.id = 1;
    current.id = 50;
    slam::Vec6 motion;
    motion[1] = 0.04;
    motion[3] = 0.12;
    motion[4] = -0.03;
    const slam::Pose truth = slam::expSE3(motion);
    std::vector<slam::Match> matches;
    // Deliberately put invalid input before valid matches to test index bookkeeping.
    matches.push_back({-1, 0, 0});
    std::mt19937 generator(42);
    for (int i = 0; i < 40; ++i) {
        const slam::Vec3 source =
            slam::makeVec3((i % 8 - 3.5) * 0.22, (i / 8 - 2) * 0.19, 2.0 + (i % 3) * 0.2);
        const slam::Vec3 target = slam::transformPoint(truth, source);
        slam::Keypoint a, b;
        a.u = static_cast<float>(camera.fx * source[0] / source[2] + camera.cx);
        a.v = static_cast<float>(camera.fy * source[1] / source[2] + camera.cy);
        b.u = static_cast<float>(camera.fx * target[0] / target[2] + camera.cx);
        b.v = static_cast<float>(camera.fy * target[1] / target[2] + camera.cy);
        candidate.keypoints.push_back(a);
        current.keypoints.push_back(b);
        candidate.depths.push_back(static_cast<float>(source[2]));
        current.depths.push_back(static_cast<float>(target[2]));
        slam::Descriptor descriptor;
        for (auto &byte : descriptor)
            byte = static_cast<std::uint8_t>(generator());
        candidate.descriptors.push_back(descriptor);
        current.descriptors.push_back(descriptor);
        matches.push_back({i, i, 0});
    }
    current.depths[4] = 0.0f;
    current.depths[9] = std::numeric_limits<float>::quiet_NaN();
    current.keypoints[15].u += 90.0f;
    for (const auto mode :
         {slam::ExecutionMode::Serial, slam::ExecutionMode::SerialSimd, slam::ExecutionMode::Parallel}) {
        const auto result = slam::verifyLoopGeometry(current, candidate, matches, camera, mode);
        check(result.success && result.inlierCount == 37, "geometric loop verification");
        check(result.inliers.size() == matches.size(), "loop inliers preserve input indexing");
        check(!result.inliers[0] && !result.inliers[5] && !result.inliers[10] && !result.inliers[16],
              "invalid matches and outliers rejected");
        checkPose(result.T_current_candidate, truth, 1e-6);
        check(result.meanError < 1e-6, "loop residual");
    }
    const auto upstream =
        ::match_frames(slam::toTrackingFeatures(current), slam::toTrackingFeatures(candidate));
    const auto matched = slam::matchFrames(current, candidate, slam::ExecutionMode::Serial);
    check(matched.size() == upstream.size() && matched.size() == 40, "reuse upstream matching");
    for (const auto &match : matched)
        check(upstream.at(static_cast<std::size_t>(match.queryIdx)) ==
                  static_cast<std::size_t>(match.trainIdx),
              "match direction");
    close(slam::scoreKeyframeSimilarity(current, candidate), 1.0, 1e-12, "upstream similarity score");
}

} // namespace

int main() {
    try {
        featureConversion();
        poseConversion();
        loopVerification();
        std::cout << "mapping adapter tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
