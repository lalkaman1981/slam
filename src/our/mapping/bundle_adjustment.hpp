#pragma once

// Local bundle adjustment (ORB-SLAM Sec. VI-D and appendix) and structure-only refinement.
// Own Levenberg-Marquardt solver with a Schur complement on the points; no g2o / Ceres.
//
// Residual of one observation (ORB-SLAM2 RGB-D convention, third row only with valid depth):
//   e = [u, v, u_R]_measured - [fx*x/z + cx, fy*y/z + cy, fx*x/z + cx - bf/z]
// Cost: sum of Huber(e^T * Omega * e), Omega = 1 / sigma_level^2 * I.

#include <vector>

#include "map.hpp"

namespace slam {

struct BAObservation {
    std::size_t poseIndex = 0;
    std::size_t pointIndex = 0;
    Vec3 measured{}; // (u, v, u_right)
    bool hasDepth = false;
    double information = 1.0; // inverse sigma^2 of the keypoint's pyramid level
    int keyframeId = -1;
    int landmarkId = -1;
    bool active = true;
};

struct BAProblem {
    CameraIntrinsics camera;
    double chi2Mono = 5.991;
    double chi2Stereo = 7.815;
    std::vector<int> keyframeIds;
    std::vector<Pose> poses; // T_cw
    std::vector<bool> fixedPose;
    std::vector<int> landmarkIds;
    std::vector<Vec3> points; // world coordinates
    std::vector<BAObservation> observations;
};

struct ObservationResidual {
    Vec3 error{};
    Mat36 jacobianPose{}; // d error / d (left SE(3) increment), twist order [omega, upsilon]
    Mat3 jacobianPoint{}; // d error / d point
    double chi2 = 0.0;
    bool inFront = false;
};

struct BAResult {
    int optimizedKeyframes = 0;
    int fixedKeyframes = 0;
    int optimizedPoints = 0;
    int observations = 0;
    double initialCost = 0.0;
    double finalCost = 0.0;
    int outliersRemoved = 0;
};

struct LandmarkRefinement {
    int landmarkId = -1;
    Vec3 position{};
    double initialChi2 = 0.0;
    double finalChi2 = 0.0;
    bool improved = false;
};

// Residuals (and Jacobians) for a batch of observations. SIMD target for the serial_simd version.
ObservationResidual evaluateObservation(const Pose &T_cw, const Vec3 &point, const BAObservation &obs,
                                        const CameraIntrinsics &camera, bool withJacobians);
void computeObservationResiduals(const BAProblem &problem, std::vector<ObservationResidual> &out,
                                 bool withJacobians, ExecutionMode mode);

// Builds the local window of Sec. VI-D: the keyframe, its covisible keyframes and all their
// points are optimized; other keyframes observing those points are included but fixed.
BAProblem buildLocalBAProblem(const Map &map, int keyframeId);
// Runs Levenberg-Marquardt on the problem. Returns the final (robust) cost.
double optimizeBAProblem(BAProblem &problem, int iterations, bool robust, ExecutionMode mode);

BAResult localBundleAdjustment(Map &map, int keyframeId, ExecutionMode mode);

// Plan functions: refine one point with fixed poses / refine independent points.
// refineLandmark only computes; refineLandmarks writes results to the map after all are done.
LandmarkRefinement refineLandmark(const Map &map, int landmarkId, ExecutionMode mode = ExecutionMode::Serial);
int refineLandmarks(Map &map, const LocalMap &localMap, ExecutionMode mode);

// Mean pixel reprojection error over all observations of the given landmarks.
double meanReprojectionError(const Map &map, const std::vector<int> &landmarkIds);

} // namespace slam
