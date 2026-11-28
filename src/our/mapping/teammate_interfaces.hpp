#pragma once

// Adapters to persons 1 and 2. Matching, scoring, rigid alignment and RANSAC
// delegate to their unchanged implementations in src/our. Mapping keeps its
// compact keyframe storage and explicitly converts pose and feature conventions.

#include <span>
#include <vector>

#include "slam_types.hpp"

namespace slam {

// ---- Person 1 ----
int hammingDistance(const Descriptor &a, const Descriptor &b);
std::vector<Match> matchFrames(const Frame &a, const Frame &b, ExecutionMode mode);
// Descriptor-overlap score in [0, 1], supplied by person 1 (no learned BoW).
double scoreKeyframeSimilarity(const Frame &a, const Frame &b);

// ---- Person 2 ----
Vec3 backProject(double u, double v, double depth, const CameraIntrinsics &camera);
bool project(const Vec3 &pointCamera, const CameraIntrinsics &camera, double &u, double &v);
Pose estimateRigidTransform(const std::vector<Vec3> &source, const std::vector<Vec3> &target);

struct LoopGeometryResult {
    bool success = false;
    Pose T_current_candidate;  // maps candidate-camera points into the current camera
    std::vector<bool> inliers; // per input match
    int inlierCount = 0;
    double meanError = 0.0; // mean 3D alignment error of inliers, metres
};

LoopGeometryResult verifyLoopGeometry(const Frame &current, const Frame &candidate,
                                      const std::vector<Match> &matches, const CameraIntrinsics &camera,
                                      ExecutionMode mode);

// Loop hypotheses use existing map points expressed in each keyframe's camera.
// This keeps triangulated points usable even when their image depth sample is absent.
LoopGeometryResult verifyLoopGeometry(std::span<const Vec3> currentPoints,
                                      std::span<const Vec3> candidatePoints, ExecutionMode mode);

} // namespace slam
