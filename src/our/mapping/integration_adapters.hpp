#pragma once

#include "../pose_estimation.hpp"
#include "slam_types.hpp"

namespace slam {

// Mapping keeps compact, image-independent keyframes. Tracking owns images and uses
// level-local integer pixels and angles in radians; mapping uses level-0 pixels and degrees.
Descriptor toMappingDescriptor(const brief_desc_t &descriptor);
brief_desc_t toTrackingDescriptor(const Descriptor &descriptor);
std::vector<ORB_Descriptor> toTrackingFeatures(const Frame &frame);
Frame toMappingFrame(const ::Frame &frame, int id, double timestamp, const CameraIntrinsics &camera);

Pose toMappingPose(const cv::Affine3d &pose);
cv::Affine3d toTrackingPose(const Pose &pose);
cv::Matx33d toCameraMatrix(const CameraIntrinsics &camera);

} // namespace slam
