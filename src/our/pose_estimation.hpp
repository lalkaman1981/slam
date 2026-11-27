#pragma once
#ifndef TEMPLATE_POSE_ESTIMATION_H
#define TEMPLATE_POSE_ESTIMATION_H

// Conventions:
//   Pose maps p_out = R p_in + t. T_cw = world -> camera (same as OpenCV's rvec/tvec).
//   Twist = (rho, phi): translation first, rotation last. Update: T <- expSE3(delta) * T.
//   Residual = predicted - observed, pixels of pyramid level 0. Distances in metres.
// "cv:" lines name the OpenCV function to compare against (tests only, not for use here).

#include "feature_extractor.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/core/affine.hpp>

// ---------------------------------------------------------------- types

// Pose = cv::Affine3d: rotation() = R (cv::Matx33d), translation() = t (cv::Vec3d); default = identity.
// Containers only: Affine3d's *, inv(), rvec() and rvec constructor are oracles for tests,
// here composePose / inversePose / expSE3 / logSE3 are our own.
// Camera = cv::Matx33d K = [fx 0 cx; 0 fy cy; 0 0 1], no distortion.

using Jacobian26 = cv::Matx<double, 2, 6>;  // d(u, v) / d(rho, phi)

struct Frame {
	std::vector<ORB_Descriptor> features;  // cv: std::vector<cv::KeyPoint> (pixels of level Candidate.level)
	cv::Mat depth;                         // CV_16UC1, raw sensor units
};

// Same 3D point seen from camera A and camera B; cv: src[i], dst[i] of estimateAffine3D
struct PointPair3D {
	cv::Point3d a;
	cv::Point3d b;
	size_t idxA=0;                         // feature index in frame A; cv::DMatch::queryIdx
	size_t idxB=0;                         // feature index in frame B; cv::DMatch::trainIdx
};

// cv: objectPoints[i], imagePoints[i] of solvePnP
struct MapObservation {
	cv::Point3d pointWorld;
	cv::Point2d pixel;                     // level 0
	double information=1.0;                // 1 / sigma^2 = 1.2^(-2 level)
};

using Sample = std::array<size_t, 3>;       // indices into the pairs
using InlierMask = std::vector<uint8_t>;    // 1 = inlier; cv: CV_8U mask, cv::Mat(mask)

struct RansacResult {
	cv::Affine3d T_ba;                     // b = R a + t
	InlierMask inliers;
	size_t numInliers=0;
};

// (H + lambda diag H) delta = b, with b = -sum w J^T r
struct PoseSystem {
	cv::Matx66d H;
	cv::Vec6d b;
	double cost=0.0;                       // sum of Huber costs
	size_t numValid=0;                     // points in front of the camera
};

struct RefineResult {
	cv::Affine3d T_cw;
	double initialRms=0.0;                 // pixels
	double finalRms=0.0;
	size_t numInliers=0;
};

enum class TrackingStatus { OK, LOST };

struct TrackResult {
	TrackingStatus status=TrackingStatus::LOST;
	cv::Affine3d T_cw;                     // meaningful only if status == OK
	orb_matches_t inlierMatches;           // geometrically verified: reference idx -> frame idx
};

// One keyframe to try in relocalizeFrame (filled by person 3)
struct RelocalizationCandidate {
	Frame keyframe;
	cv::Affine3d T_cw;
	orb_matches_t matches;                 // keyframe idx -> frame idx
};

// ---------------------------------------------------------------- parameters

constexpr double DEPTH_SCALE = 5000.0;          // raw units per metre (TUM)
constexpr double MIN_DEPTH = 0.1;               // metres
constexpr double MAX_DEPTH = 8.0;
constexpr double MIN_SAMPLE_SPACING = 0.05;     // metres
constexpr double MIN_TRIANGLE_AREA = 1e-4;      // m^2
constexpr double RANSAC_THRESHOLD = 0.03;       // metres
constexpr int RANSAC_MAX_ITERATIONS = 500;
constexpr double RANSAC_CONFIDENCE = 0.99;
constexpr uint64_t RANSAC_SEED = 42;
constexpr size_t MIN_INLIERS = 15;
constexpr double HUBER_DELTA = 2.4477;          // sqrt(5.991): chi^2, 2 dof, 95%
constexpr int REFINE_MAX_ITERATIONS = 10;

// ---------------------------------------------------------------- SE(3), shared with person 3

// A*B: R = Ra Rb, t = Ra tb + ta
// cv: cv::Affine3d A * B; cv::composeRT(B, A) -- arguments reversed, its first transform is applied first
cv::Affine3d composePose(const cv::Affine3d& A, const cv::Affine3d& B);

// R^T, -R^T t
// cv: cv::Affine3d::inv()
cv::Affine3d inversePose(const cv::Affine3d& T);

// R = exp([phi]x), t = V(phi) rho
// cv: cv::Rodrigues(phi) for R only; tvec is t itself, NOT rho
cv::Affine3d expSE3(const cv::Vec6d& xi);

// phi = log(R), rho = V(phi)^-1 t; inverse of expSE3 for |phi| < pi
// cv: cv::Rodrigues(R) for phi only
cv::Vec6d logSE3(const cv::Affine3d& T);

// ---------------------------------------------------------------- camera

// d = raw / DEPTH_SCALE; nullopt if outside the image, raw == 0, or d outside [MIN_DEPTH, MAX_DEPTH]
// cv: cv::rgbd::rescaleDepth (contrib; whole image, marks invalid as NaN)
std::optional<double> readValidDepth(const Frame& frame, const cv::Point& pixel);

// X = (u - cx) Z / fx, Y = (v - cy) Z / fy, Z = depth
// cv: cv::rgbd::depthTo3dSparse (contrib); cv::undistortPoints gives (X/Z, Y/Z) only
cv::Point3d backProject(const cv::Point2d& pixel, double depth, const cv::Matx33d& K);

// u = fx X/Z + cx, v = fy Y/Z + cy; nullopt if Z <= 0
// cv: cv::projectPoints with rvec = tvec = 0 (it does not check Z > 0)
std::optional<cv::Point2d> project(const cv::Point3d& pointCamera, const cv::Matx33d& K);

// out[i] = R points[i] + t
// cv: cv::transform(src, dst, 3x4 [R|t])
std::vector<cv::Point3d> transformPoints(std::span<const cv::Point3d> points, const cv::Affine3d& pose);

// ---------------------------------------------------------------- 3D-3D pose

// level-0 pixel = coords * 1.2^level -> round -> depth -> back-project; drops pairs without depth; sort by idxA
// cv: cv::rgbd::depthTo3dSparse on both frames + filtering by hand; matches = std::vector<cv::DMatch>
std::vector<PointPair3D> build3DCorrespondences(const Frame& a, const Frame& b,
	const orb_matches_t& matches, const cv::Matx33d& K);

// In A or B: some |pi - pj| < MIN_SAMPLE_SPACING, or |(p1-p0) x (p2-p0)| / 2 < MIN_TRIANGLE_AREA
// cv: none public (internal checkSubset of estimateAffine3D)
bool isDegenerateSample(std::span<const PointPair3D, 3> sample);

// Horn: N (4x4) from S = sum a' b'^T, q = eigenvector of the largest eigenvalue, t = cb - R ca
// nullopt if fewer than 3 pairs or the rotation is ambiguous
// cv: cv::estimateAffine3D(src, dst, nullptr, true) (Umeyama/SVD, scale = 1); uses cv::eigen for N
std::optional<cv::Affine3d> estimateRigidTransform(std::span<const PointPair3D> pairs);

// 3 distinct indices < numPairs (numPairs >= 3), depending only on (seed, iteration)
// cv: none public (internal getSubset); cv::RNG for random numbers
Sample sampleRansacHypothesis(uint64_t seed, int iteration, size_t numPairs);

// errors[i] = |R a_i + t - b_i|
// cv: cv::transform, then cv::norm per row (internal computeError of estimateAffine3D)
std::vector<double> computeAlignmentErrors(std::span<const PointPair3D> pairs, const cv::Affine3d& T_ba);

// mask[i] = errors[i] <= threshold
// cv: cv::compare(errors, threshold, mask, cv::CMP_LE) (writes 255, not 1)
InlierMask selectInliers(std::span<const double> errors, double threshold);

// sample -> degenerate? -> Horn -> errors -> mask -> keep best -> refit on its inliers
// nullopt if the best hypothesis has fewer than MIN_INLIERS
// cv: cv::estimateAffine3D(src, dst, out, inliers, threshold, confidence) -- fits a 12-dof affine, not rigid
std::optional<RansacResult> estimatePoseRansac(std::span<const PointPair3D> pairs,
	double threshold=RANSAC_THRESHOLD,
	int maxIterations=RANSAC_MAX_ITERATIONS,
	double confidence=RANSAC_CONFIDENCE,
	uint64_t seed=RANSAC_SEED);

// ---------------------------------------------------------------- 3D-2D refinement

// mapMatches: frame feature idx -> index in mapPoints (from match_frame2map)
// cv: none; fills objectPoints / imagePoints for solvePnP
std::vector<MapObservation> buildMapObservations(const Frame& frame,
	std::span<const cv::Point3d> mapPoints, const orb_matches_t& mapMatches);

// r_i = project(T_cw P_i) - uv_i; NaN if the point is behind the camera
// cv: cv::projectPoints, then subtract imagePoints
std::vector<cv::Point2d> computeReprojectionResiduals(std::span<const MapObservation> observations,
	const cv::Affine3d& T_cw, const cv::Matx33d& K);

// J_i = d project/d pc * [I | -[pc]x], pc = T_cw P_i
// cv: cv::projectPoints(pc, rvec = 0, tvec = 0, ..., jacobian): cols 0-2 = phi, 3-5 = rho (ours: rho first)
std::vector<Jacobian26> computePoseJacobians(std::span<const MapObservation> observations,
	const cv::Affine3d& T_cw, const cv::Matx33d& K);

// H = sum w info J^T J, b = -sum w info J^T r; e = sqrt(info) |r|, w = min(1, HUBER_DELTA / e); skips NaN r
// cv: none public (solvePnPRefineLM does it internally, without robust weights)
PoseSystem accumulatePoseSystem(std::span<const cv::Point2d> residuals,
	std::span<const Jacobian26> jacobians, std::span<const MapObservation> observations);

// (H + lambda diag H) delta = b by LDL^T; nullopt if a pivot is ~0
// cv: cv::solve(H, b, delta, cv::DECOMP_CHOLESKY) (returns false if singular)
std::optional<cv::Vec6d> solvePoseIncrement(const cv::Matx66d& H, const cv::Vec6d& b, double lambda=0.0);

// repeat: residuals, jacobians, H b, solve, T <- exp(delta) T; on cost increase reject and raise lambda
// nullopt if the system is degenerate
// cv: cv::solvePnPRefineLM (no Huber)
std::optional<RefineResult> refinePose(std::span<const MapObservation> observations,
	const cv::Affine3d& initialT_cw, const cv::Matx33d& K, int maxIterations=REFINE_MAX_ITERATIONS);

// ---------------------------------------------------------------- tracking

// matches: reference idx -> frame idx. RANSAC -> T_cw = T_cr T_rw -> refinePose on map observations -> OK / LOST
// cv: cv::rgbd::RgbdOdometry::compute (contrib; dense); with a map: cv::solvePnPRansac + solvePnPRefineLM
TrackResult trackFrame(const Frame& frame, const Frame& reference, const cv::Affine3d& referenceT_rw,
	const orb_matches_t& matches, std::span<const cv::Point3d> mapPoints,
	const orb_matches_t& mapMatches, const cv::Matx33d& K);

// matches: candidate idx -> current idx. T_ba = candidate -> current, nullopt if rejected
// cv: cv::estimateAffine3D (RANSAC) on depth; without depth: cv::findEssentialMat + cv::recoverPose
std::optional<RansacResult> verifyLoopGeometry(const Frame& current, const Frame& candidate,
	const orb_matches_t& matches, const cv::Matx33d& K);

// Candidate with the most verified inliers: T_cw = T_ck T_kw; LOST if none passes
// cv: cv::solvePnPRansac per candidate
TrackResult relocalizeFrame(const Frame& frame,
	std::span<const RelocalizationCandidate> candidates, const cv::Matx33d& K);

#endif //TEMPLATE_POSE_ESTIMATION_H
