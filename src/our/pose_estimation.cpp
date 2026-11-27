#include "pose_estimation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#define DEFAULT_SCALE_FACTOR 1.2

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

// d = raw / DEPTH_SCALE; nullopt if outside the image, raw == 0, or d outside [MIN_DEPTH, MAX_DEPTH]
std::optional<double> readValidDepth(const Frame& frame, const cv::Point& pixel) {
	const cv::Rect image(0, 0, frame.depth.cols, frame.depth.rows);
	if (!image.contains(pixel)) return std::nullopt;

	const uint16_t raw = frame.depth.at<uint16_t>(pixel);  // at(Point) = (row y, col x)
	if (raw == 0) return std::nullopt;

	const double d = raw / DEPTH_SCALE;
	if (d < MIN_DEPTH || d > MAX_DEPTH) return std::nullopt;
	return d;
}

cv::Point3d backProject(const cv::Point2d& pixel, double depth, const cv::Matx33d& K) {
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

// out[i] = R points[i] + t
std::vector<cv::Point3d> transformPoints(std::span<const cv::Point3d> points, const cv::Affine3d& pose) {
	const cv::Matx33d R = pose.rotation();
	const cv::Vec3d t = pose.translation();
	std::vector<cv::Point3d> result;
	result.reserve(points.size());
	for (const auto& p : points) {
		const cv::Vec3d q = R * cv::Vec3d(p) + t;
		result.emplace_back(q[0], q[1], q[2]);
	}
	return result;
}

// ---------------------------------------------------------------- 3D-3D pose

// level-0 pixel = round(coords * 1.2^level)
static cv::Point levelZeroPixel(const Candidate& c, double scale_factor=DEFAULT_SCALE_FACTOR) {
	const auto [x, y] = scale_coords(c.coords, std::pow(scale_factor, c.level));
	return {x, y};
}
// pair (i,j): level-0 pixel -> depth -> back-project, in both frames; sorted by idxA
std::vector<PointPair3D> build3DCorrespondences(const Frame& a, const Frame& b,
	const orb_matches_t& matches, const cv::Matx33d& K)
{
	std::vector<PointPair3D> pairs;
	pairs.reserve(matches.size());
	for (const auto& [i, j] : matches) {
		const cv::Point pa = levelZeroPixel(a.features[i].candidate);
		const cv::Point pb = levelZeroPixel(b.features[j].candidate);
		const auto za = readValidDepth(a, pa);
		const auto zb = readValidDepth(b, pb);
		if (!za || !zb) continue;
		pairs.push_back({backProject(pa, *za, K), backProject(pb, *zb, K), i, j});
	}
	std::ranges::sort(pairs, {}, &PointPair3D::idxA);
	return pairs;
}

// any |pi - pj| < MIN_SAMPLE_SPACING or |(p1-p0) x (p2-p0)| / 2 < MIN_TRIANGLE_AREA
static bool isDegenerateTriangle(const cv::Point3d& p0, const cv::Point3d& p1, const cv::Point3d& p2) {
	const bool tooClose = cv::norm(p1 - p0) < MIN_SAMPLE_SPACING || cv::norm(p2 - p0) < MIN_SAMPLE_SPACING
	                   || cv::norm(p2 - p1) < MIN_SAMPLE_SPACING;
	return tooClose || cv::norm((p1 - p0).cross(p2 - p0)) / 2.0 < MIN_TRIANGLE_AREA;
}

bool isDegenerateSample(std::span<const PointPair3D, 3> sample) {
	return isDegenerateTriangle(sample[0].a, sample[1].a, sample[2].a)
	    || isDegenerateTriangle(sample[0].b, sample[1].b, sample[2].b);
}

// unit q = (w, x, y, z) -> R
static cv::Matx33d quaternionToRotation(double w, double x, double y, double z) {
	return {1 - 2 * (y * y + z * z),     2 * (x * y - w * z),     2 * (x * z + w * y),
	            2 * (x * y + w * z), 1 - 2 * (x * x + z * z),     2 * (y * z - w * x),
	            2 * (x * z - w * y),     2 * (y * z + w * x), 1 - 2 * (x * x + y * y)};
}

// Horn: N (4x4) from S = sum a' b'^T, q = eigenvector of the largest eigenvalue, t = cb - R ca
std::optional<cv::Affine3d> estimateRigidTransform(std::span<const PointPair3D> pairs) {
	if (pairs.size() < 3) return std::nullopt;

	cv::Point3d ca, cb;
	for (const auto& p : pairs) { ca += p.a; cb += p.b; }
	ca *= 1.0 / pairs.size();
	cb *= 1.0 / pairs.size();

	cv::Matx33d S = cv::Matx33d::zeros();
	for (const auto& p : pairs)
		S += cv::Vec3d(p.a - ca) * cv::Vec3d(p.b - cb).t();

	const double xx = S(0,0), xy = S(0,1), xz = S(0,2);
	const double yx = S(1,0), yy = S(1,1), yz = S(1,2);
	const double zx = S(2,0), zy = S(2,1), zz = S(2,2);
	const cv::Matx44d N(xx + yy + zz, yz - zy,       zx - xz,       xy - yx,
	              yz - zy,      xx - yy - zz,  xy + yx,       zx + xz,
	              zx - xz,      xy + yx,      -xx + yy - zz,  yz + zy,
	              xy - yx,      zx + xz,       yz + zy,      -xx - yy + zz);

	cv::Matx41d lambda;
	cv::Matx44d E;                                  // row i = eigenvector of lambda(i), descending
	cv::eigen(N, lambda, E);
	if (lambda(0) - lambda(1) <= 1e-10 * std::abs(lambda(0))) return std::nullopt;  // collinear: rotation about the line is free
	const cv::Matx33d R = quaternionToRotation(E(0, 0), E(0, 1), E(0, 2), E(0, 3));
	const cv::Vec3d t = cv::Vec3d(cb) - R * cv::Vec3d(ca);
	return cv::Affine3d(R, t);
}

// splitmix64: a well-mixed 64-bit hash of a counter
static uint64_t splitmix64(uint64_t x) {
	x += 0x9E3779B97F4A7C15ull;
	x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
	x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
	return x ^ (x >> 31);
}

// 3 distinct indices from the counter stream splitmix64(seed, iteration, k)
Sample sampleRansacHypothesis(uint64_t seed, int iteration, size_t numPairs) {
	uint64_t state = splitmix64(seed) ^ static_cast<uint64_t>(iteration);
	Sample s{};
	for (size_t n = 0; n < 3;) {
		s[n] = splitmix64(state++) % numPairs;  // shortcut: modulo bias ~numPairs / 2^64, negligible
		if (std::find(s.begin(), s.begin() + n, s[n]) == s.begin() + n) ++n;
	}
	return s;
}

// errors[i] = |R a_i + t - b_i|
std::vector<double> computeAlignmentErrors(std::span<const PointPair3D> pairs, const cv::Affine3d& T_ba) {
	const cv::Matx33d R = T_ba.rotation();
	const cv::Vec3d t = T_ba.translation();
	std::vector<double> errors;
	errors.reserve(pairs.size());
	for (const auto& p : pairs)
		errors.push_back(cv::norm(R * cv::Vec3d(p.a) + t - cv::Vec3d(p.b)));
	return errors;
}

// mask[i] = errors[i] <= threshold
InlierMask selectInliers(std::span<const double> errors, double threshold) {
	InlierMask mask;
	mask.reserve(errors.size());
	for (const double e : errors) mask.push_back(e <= threshold);
	return mask;
}

static size_t countInliers(const InlierMask& mask) {
	return std::ranges::count(mask, 1);
}

// k = log(1 - confidence) / log(1 - w^3), w = inlier ratio
static int requiredIterations(size_t numInliers, size_t numPairs, double confidence) {
	const double w3 = std::pow(static_cast<double>(numInliers) / numPairs, 3);
	if (w3 >= 1.0) return 0;
	if (w3 <= 0.0) return std::numeric_limits<int>::max();
	return static_cast<int>(std::ceil(std::log(1.0 - confidence) / std::log(1.0 - w3)));
}

// sample -> degenerate? -> Horn -> errors -> mask -> keep best -> refit on its inliers
std::optional<RansacResult> estimatePoseRansac(std::span<const PointPair3D> pairs,
	double threshold,
	int maxIterations,
	double confidence,
	uint64_t seed) {
	if (pairs.size() < 3) return std::nullopt;

	std::optional<RansacResult> best;
	int limit = maxIterations;
	for (int it = 0; it < limit; ++it) {
		const Sample s = sampleRansacHypothesis(seed, it, pairs.size());
		const std::array<PointPair3D, 3> sample{pairs[s[0]], pairs[s[1]], pairs[s[2]]};
		if (isDegenerateSample(sample)) continue;
		const auto T = estimateRigidTransform(sample);
		if (!T) continue;

		InlierMask mask = selectInliers(computeAlignmentErrors(pairs, *T), threshold);
		const size_t n = countInliers(mask);
		if (best && n <= best->numInliers) continue;
		best = RansacResult{*T, std::move(mask), n};
		limit = std::min(maxIterations, requiredIterations(n, pairs.size(), confidence));
	}
	if (!best || best->numInliers < MIN_INLIERS) return std::nullopt;

	std::vector<PointPair3D> inliers;
	for (size_t i = 0; i < pairs.size(); ++i)
		if (best->inliers[i]) inliers.push_back(pairs[i]);
	if (const auto T = estimateRigidTransform(inliers)) {
		InlierMask mask = selectInliers(computeAlignmentErrors(pairs, *T), threshold);
		const size_t n = countInliers(mask);
		if (n >= best->numInliers) *best = RansacResult{*T, std::move(mask), n};
	}
	return best;
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
