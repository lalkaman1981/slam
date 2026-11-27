#include "integration_adapters.hpp"

#include <cmath>
#include <cstring>
#include <numbers>
#include <stdexcept>

namespace slam {

Descriptor toMappingDescriptor(const brief_desc_t &descriptor) {
    static_assert(sizeof(brief_desc_t) == sizeof(Descriptor));
    Descriptor result;
    std::memcpy(result.data(), descriptor.data(), sizeof(result));
    return result;
}

brief_desc_t toTrackingDescriptor(const Descriptor &descriptor) {
    brief_desc_t result;
    std::memcpy(result.data(), descriptor.data(), sizeof(result));
    return result;
}

std::vector<ORB_Descriptor> toTrackingFeatures(const Frame &frame) {
    if (frame.keypoints.size() != frame.descriptors.size())
        throw std::invalid_argument("keypoints and descriptors must have the same size");
    std::vector<ORB_Descriptor> features;
    features.reserve(frame.keypoints.size());
    for (std::size_t i = 0; i < frame.keypoints.size(); ++i) {
        const Keypoint &point = frame.keypoints[i];
        const double scale = std::pow(1.3, point.octave);
        Candidate candidate{
            point.response,
            {static_cast<int>(std::lround(point.u / scale)), static_cast<int>(std::lround(point.v / scale))},
            point.octave};
        features.push_back({candidate, toTrackingDescriptor(frame.descriptors[i]),
                            point.angle * static_cast<float>(std::numbers::pi / 180.0)});
    }
    return features;
}

Frame toMappingFrame(const ::Frame &frame, int id, double timestamp, const CameraIntrinsics &camera) {
    if (!frame.depth.empty() && (frame.depth.type() != CV_16UC1 || frame.depth.cols != camera.width ||
                                 frame.depth.rows != camera.height))
        throw std::invalid_argument("depth must be registered CV_16UC1 at the camera image size");
    Frame result;
    result.id = id;
    result.timestamp = timestamp;
    result.keypoints.reserve(frame.features.size());
    result.descriptors.reserve(frame.features.size());
    result.depths.reserve(frame.features.size());
    for (const ORB_Descriptor &feature : frame.features) {
        // Depth is sampled at rounded pixels; bundle adjustment retains the subpixel
        // location, matching person 2's buildMapObservations.
        const double scale = std::pow(1.3, feature.candidate.level);
        const auto [u, v] = ::scale_coords(feature.candidate.coords, scale);
        result.keypoints.push_back(
            {static_cast<float>(feature.candidate.coords.first * scale),
             static_cast<float>(feature.candidate.coords.second * scale), feature.candidate.level,
             feature.orientation * static_cast<float>(180.0 / std::numbers::pi), feature.candidate.score});
        result.descriptors.push_back(toMappingDescriptor(feature.descriptor));
        const auto depth = ::readValidDepth(frame, {u, v});
        result.depths.push_back(depth ? static_cast<float>(*depth) : 0.0f);
    }
    return result;
}

Pose toMappingPose(const cv::Affine3d &pose) {
    const cv::Matx33d rotation = pose.rotation();
    const cv::Vec3d translation = pose.translation();
    Pose result;
    for (int row = 0; row < 3; ++row) {
        result.t[static_cast<std::size_t>(row)] = translation[row];
        for (int col = 0; col < 3; ++col)
            result.R(static_cast<std::size_t>(row), static_cast<std::size_t>(col)) = rotation(row, col);
    }
    return result;
}

cv::Affine3d toTrackingPose(const Pose &pose) {
    cv::Matx33d rotation;
    cv::Vec3d translation;
    for (int row = 0; row < 3; ++row) {
        translation[row] = pose.t[static_cast<std::size_t>(row)];
        for (int col = 0; col < 3; ++col)
            rotation(row, col) = pose.R(static_cast<std::size_t>(row), static_cast<std::size_t>(col));
    }
    return {rotation, translation};
}

cv::Matx33d toCameraMatrix(const CameraIntrinsics &camera) {
    return {camera.fx, 0.0, camera.cx, 0.0, camera.fy, camera.cy, 0.0, 0.0, 1.0};
}

} // namespace slam
