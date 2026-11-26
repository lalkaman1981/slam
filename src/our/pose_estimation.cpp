#include "pose_estimation.hpp"

#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------- SE(3), shared with person 3

// A(B(p)) = Ra (Rb p + tb) + ta
cv::Affine3d composePose(const cv::Affine3d& A, const cv::Affine3d& B) {
	const cv::Matx33d Ra = A.rotation();
	return {Ra * B.rotation(), Ra * B.translation() + A.translation()};
}

// q = R p + t  =>  p = R^T q - R^T t
cv::Affine3d inversePose(const cv::Affine3d& T) {
	const cv::Matx33d Rt = T.rotation().t();
	return {Rt, -(Rt * T.translation())};
}

// [phi]x v = phi x v
static cv::Matx33d skew(const cv::Vec3d& phi) {
	return {      0, -phi[2],  phi[1],
	         phi[2],       0, -phi[0],
	        -phi[1],  phi[0],       0};
}

// Below this angle the formulas approach 0/0, so Taylor series are used instead.
constexpr double SMALL_ANGLE = 1e-3;

// a = sin(theta) / theta
static double sinCoefficient(double theta) {
	return theta < SMALL_ANGLE ? 1.0 - theta * theta / 6.0 : std::sin(theta) / theta;
}

// b = (1 - cos(theta)) / theta^2 = (sin(theta/2) / (theta/2))^2 / 2
static double cosCoefficient(double theta) {
	const double s = sinCoefficient(theta / 2.0);
	return s * s / 2.0;
}

// R = I + a [phi]x + b [phi]x^2,  V = I + b [phi]x + c [phi]x^2,  t = V rho
cv::Affine3d expSE3(const cv::Vec6d& xi) {
	const cv::Vec3d rho(xi[0], xi[1], xi[2]);
	const cv::Vec3d phi(xi[3], xi[4], xi[5]);
	const double theta = cv::norm(phi);

	const double a = sinCoefficient(theta);
	const double b = cosCoefficient(theta);
	const double c = theta < SMALL_ANGLE ? 1.0 / 6.0 - theta * theta / 120.0
	                                     : (1.0 - a) / (theta * theta);

	const cv::Matx33d I = cv::Matx33d::eye();
	const cv::Matx33d W = skew(phi);
	const cv::Matx33d R = I + a * W + b * (W * W);
	const cv::Matx33d V = I + b * W + c * (W * W);
	return {R, V * rho};
}

// phi = theta * axis,  theta = atan2(sin, cos)
static cv::Vec3d logRotation(const cv::Matx33d& R) {
	const cv::Vec3d w(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));  // 2 sin(theta) axis
	const double sinTheta = cv::norm(w) / 2.0;
	const double cosTheta = (R(0,0) + R(1,1) + R(2,2) - 1.0) / 2.0;
	const double theta = std::atan2(sinTheta, cosTheta);

	if (cosTheta > 0.0)
		return w * (0.5 / sinCoefficient(theta));

	// Near pi w vanishes, so take the axis from (R + R^T)/2 - cos I = (1 - cos) axis axis^T.
	const cv::Matx33d S = 0.5 * (R + R.t()) - cosTheta * cv::Matx33d::eye();
	int k = 0;                              // the longest column is the most accurate
	if (S(1,1) > S(k,k)) k = 1;
	if (S(2,2) > S(k,k)) k = 2;
	const cv::Vec3d axis = cv::normalize(cv::Vec3d(S(0,k), S(1,k), S(2,k)));
	return axis * (axis.dot(w) < 0.0 ? -theta : theta);
}

// rho = V^-1 t,  V^-1 = I - [phi]x / 2 + d [phi]x^2,  d = (1 - a / (2b)) / theta^2
cv::Vec6d logSE3(const cv::Affine3d& T) {
	const cv::Vec3d phi = logRotation(T.rotation());
	const double theta = cv::norm(phi);

	const double a = sinCoefficient(theta);
	const double b = cosCoefficient(theta);
	const double d = theta < SMALL_ANGLE ? 1.0 / 12.0 + theta * theta / 720.0
	                                     : (1.0 - a / (2.0 * b)) / (theta * theta);

	const cv::Matx33d W = skew(phi);
	const cv::Matx33d Vinv = cv::Matx33d::eye() - 0.5 * W + d * (W * W);
	const cv::Vec3d rho = Vinv * T.translation();
	return {rho[0], rho[1], rho[2], phi[0], phi[1], phi[2]};
}

// ---------------------------------------------------------------- camera

std::optional<double> readValidDepth(const Frame& frame, cv::Point pixel) {
	throw std::logic_error("TODO: readValidDepth");
}

cv::Point3d backProject(cv::Point2d pixel, double depth, const cv::Matx33d& K) {
	const double cx = K(0, 2);
	const double cy = K(1, 2);
	const double fx = K(0, 0);
	const double fy = K(1, 1);

	return {(pixel.x - cx) / fx * depth, (pixel.y - cy) / fy * depth, depth};
}

std::optional<cv::Point2d> project(const cv::Point3d& pointCamera, const cv::Matx33d& K) {
	constexpr double kMinDepth = 1e-9;          
	if (pointCamera.z <= kMinDepth)
		return std::nullopt;

	const double invZ = 1.0 / pointCamera.z;
	const double u = K(0,0) * pointCamera.x * invZ + K(0,1) * pointCamera.y * invZ + K(0,2);
	const double v = K(1,1) * pointCamera.y * invZ + K(1,2);
	return cv::Point2d{u, v};
}

std::vector<cv::Point3d> transformPoints(std::span<const cv::Point3d> points, const cv::Affine3d& pose) {
	throw std::logic_error("TODO: transformPoints");
}

// ---------------------------------------------------------------- 3D-3D pose

std::vector<PointPair3D> build3DCorrespondences(const Frame& a, const Frame& b,
	const orb_matches_t& matches, const cv::Matx33d& K) {
	throw std::logic_error("TODO: build3DCorrespondences");
}

bool isDegenerateSample(std::span<const PointPair3D, 3> sample) {
	throw std::logic_error("TODO: isDegenerateSample");
}

std::optional<cv::Affine3d> estimateRigidTransform(std::span<const PointPair3D> pairs) {
	throw std::logic_error("TODO: estimateRigidTransform");
}

Sample sampleRansacHypothesis(uint64_t seed, int iteration, size_t numPairs) {
	throw std::logic_error("TODO: sampleRansacHypothesis");
}

std::vector<double> computeAlignmentErrors(std::span<const PointPair3D> pairs, const cv::Affine3d& T_ba) {
	throw std::logic_error("TODO: computeAlignmentErrors");
}

InlierMask selectInliers(std::span<const double> errors, double threshold) {
	throw std::logic_error("TODO: selectInliers");
}

std::optional<RansacResult> estimatePoseRansac(std::span<const PointPair3D> pairs,
	double threshold,
	int maxIterations,
	double confidence,
	uint64_t seed) {
	throw std::logic_error("TODO: estimatePoseRansac");
}

// ---------------------------------------------------------------- 3D-2D refinement

std::vector<MapObservation> buildMapObservations(const Frame& frame,
	std::span<const cv::Point3d> mapPoints, const orb_matches_t& mapMatches) {
	throw std::logic_error("TODO: buildMapObservations");
}

std::vector<cv::Point2d> computeReprojectionResiduals(std::span<const MapObservation> observations,
	const cv::Affine3d& T_cw, const cv::Matx33d& K) {
	throw std::logic_error("TODO: computeReprojectionResiduals");
}

std::vector<Jacobian26> computePoseJacobians(std::span<const MapObservation> observations,
	const cv::Affine3d& T_cw, const cv::Matx33d& K) {
	throw std::logic_error("TODO: computePoseJacobians");
}

PoseSystem accumulatePoseSystem(std::span<const cv::Point2d> residuals,
	std::span<const Jacobian26> jacobians, std::span<const MapObservation> observations) {
	throw std::logic_error("TODO: accumulatePoseSystem");
}

std::optional<cv::Vec6d> solvePoseIncrement(const cv::Matx66d& H, const cv::Vec6d& b, double lambda) {
	throw std::logic_error("TODO: solvePoseIncrement");
}

std::optional<RefineResult> refinePose(std::span<const MapObservation> observations,
	const cv::Affine3d& initialT_cw, const cv::Matx33d& K, int maxIterations) {
	throw std::logic_error("TODO: refinePose");
}

// ---------------------------------------------------------------- tracking

TrackResult trackFrame(const Frame& frame, const Frame& reference, const cv::Affine3d& referenceT_rw,
	const orb_matches_t& matches, std::span<const cv::Point3d> mapPoints,
	const orb_matches_t& mapMatches, const cv::Matx33d& K) {
	throw std::logic_error("TODO: trackFrame");
}

std::optional<RansacResult> verifyLoopGeometry(const Frame& current, const Frame& candidate,
	const orb_matches_t& matches, const cv::Matx33d& K) {
	throw std::logic_error("TODO: verifyLoopGeometry");
}

TrackResult relocalizeFrame(const Frame& frame,
	std::span<const RelocalizationCandidate> candidates, const cv::Matx33d& K) {
	throw std::logic_error("TODO: relocalizeFrame");
}
