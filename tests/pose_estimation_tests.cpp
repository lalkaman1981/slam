// SE(3) tests; cv::Affine3d is the oracle for compose, inverse and the rotation part of exp.
#include "pose_estimation.hpp"

#include <cmath>
#include <cstdio>
#include <numbers>
#include <vector>

static int failures = 0;

static void checkSmall(double error, const char* name, double theta) {
	if (error <= 1e-9) return;
	++failures;
	std::printf("FAIL %s, theta = %g: error %g\n", name, theta, error);
}

static double distance(const cv::Affine3d& A, const cv::Affine3d& B) {
	return cv::norm(A.matrix - B.matrix);
}

int main() {
	constexpr double pi = std::numbers::pi;
	cv::RNG rng(1);
	for (double theta : {0.0, 1e-12, 1e-6, 9.99e-4, 1.001e-3, 0.1, 1.0, pi / 2, 3.0, pi - 1e-4, pi - 1e-7}) {
		for (int i = 0; i < 100; ++i) {
			const cv::Vec3d axis = cv::normalize(cv::Vec3d(rng.gaussian(1), rng.gaussian(1), rng.gaussian(1)));
			const cv::Vec3d phi = theta * axis;
			const cv::Vec6d xi(rng.gaussian(1), rng.gaussian(1), rng.gaussian(1), phi[0], phi[1], phi[2]);
			const cv::Affine3d T = expSE3(xi);
			const cv::Affine3d half = expSE3(xi * 0.5);
			const cv::Affine3d other = expSE3(cv::Vec6d(1, -2, 3, 0.3, 0.2, -0.5));

			checkSmall(cv::norm(T.rotation() - cv::Affine3d(phi).rotation()), "exp rotation", theta);
			checkSmall(cv::norm(logSE3(T) - xi), "log(exp(xi)) = xi", theta);
			checkSmall(distance(half * half, T), "exp(xi/2)^2 = exp(xi)", theta);
			checkSmall(distance(composePose(T, other), T * other), "compose", theta);
			checkSmall(distance(inversePose(T), T.inv()), "inverse", theta);
		}
	}

	// Moving along x while turning 90 degrees left: a quarter circle of radius 2/pi.
	const cv::Affine3d arc = expSE3(cv::Vec6d(1, 0, 0, 0, 0, pi / 2));
	checkSmall(cv::norm(arc.translation() - cv::Vec3d(2 / pi, 2 / pi, 0)), "quarter circle", pi / 2);

	// Exactly pi: the axis sign is free, so check the pose, not the twist.
	for (const cv::Matx33d& R : {cv::Matx33d(1, 0, 0, 0, -1, 0, 0, 0, -1), cv::Matx33d(-1, 0, 0, 0, 1, 0, 0, 0, -1),
	                             cv::Matx33d(-1, 0, 0, 0, -1, 0, 0, 0, 1)}) {
		const cv::Affine3d T(R, cv::Vec3d(1, 2, 3));
		checkSmall(distance(expSE3(logSE3(T)), T), "exp(log(T)) = T", pi);
	}

	// Horn + RANSAC on a random scene: 200 pairs, 30% shuffled outliers, cv::Affine3d as the truth.
	const cv::Affine3d T_true = expSE3(cv::Vec6d(0.1, -0.2, 0.05, 0.2, -0.1, 0.3));
	std::vector<PointPair3D> pairs;
	for (size_t i = 0; i < 200; ++i) {
		const cv::Point3d a(rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0), rng.uniform(1.0, 4.0));
		const cv::Vec3d b = T_true * cv::Vec3d(a);
		pairs.push_back({a, {b[0], b[1], b[2]}, i, i});
	}
	const auto horn = estimateRigidTransform(pairs);
	checkSmall(horn ? distance(*horn, T_true) : 1.0, "Horn exact", 0);
	for (size_t i = 0; i < 60; ++i) pairs[i].b.x += 0.5;  // outliers: 50 cm off
	const auto ransac = estimatePoseRansac(pairs);
	checkSmall(ransac ? distance(ransac->T_ba, T_true) : 1.0, "RANSAC pose", 0);
	checkSmall(ransac ? std::abs(double(ransac->numInliers) - 140) : 1.0, "RANSAC inliers", 0);

	const std::vector<PointPair3D> line{{{0, 0, 1}, {0, 0, 1}}, {{1, 0, 1}, {1, 0, 1}}, {{2, 0, 1}, {2, 0, 1}}};
	checkSmall(estimateRigidTransform(line).has_value(), "collinear -> nullopt", 0);
	checkSmall(isDegenerateSample(std::span<const PointPair3D, 3>(line.data(), 3)) ? 0 : 1, "collinear degenerate", 0);

	const Sample s = sampleRansacHypothesis(7, 3, 5);
	checkSmall(s == sampleRansacHypothesis(7, 3, 5) && s[0] != s[1] && s[0] != s[2] && s[1] != s[2]
	           && s[0] < 5 && s[1] < 5 && s[2] < 5 ? 0 : 1, "sample distinct, deterministic", 0);

	// 3D-2D refinement: 300 world points seen by a camera at T_cw, 2D noise sigma = 0.5 px.
	const cv::Matx33d K(525, 0, 319.5, 0, 525, 239.5, 0, 0, 1);
	const cv::Affine3d T_cw = expSE3(cv::Vec6d(0.05, -0.1, 0.2, 0.05, 0.1, -0.02));
	std::vector<MapObservation> observations;
	for (int i = 0; i < 300; ++i) {
		const cv::Point3d P(rng.uniform(-2.0, 2.0), rng.uniform(-1.5, 1.5), rng.uniform(2.0, 6.0));
		const cv::Vec3d pc = T_cw * cv::Vec3d(P);
		const cv::Point2d uv = *project({pc[0], pc[1], pc[2]}, K);
		observations.push_back({P, uv + cv::Point2d(rng.gaussian(0.5), rng.gaussian(0.5)), 1.0});
	}

	// Jacobian = central differences of the residual under T <- exp(eps e_k) T
	const auto J = computePoseJacobians(observations, T_cw, K);
	double jacobianError = 0.0;
	for (int k = 0; k < 6; ++k) {
		cv::Vec6d eps = cv::Vec6d::zeros();
		eps[k] = 1e-6;
		const auto rp = computeReprojectionResiduals(observations, composePose(expSE3(eps), T_cw), K);
		const auto rm = computeReprojectionResiduals(observations, composePose(expSE3(-eps), T_cw), K);
		for (size_t i = 0; i < observations.size(); ++i) {
			const cv::Point2d numeric = (rp[i] - rm[i]) * (0.5 / eps[k]);
			jacobianError = std::max({jacobianError, std::abs(numeric.x - J[i](0, k)), std::abs(numeric.y - J[i](1, k))});
		}
	}
	checkSmall(jacobianError * 1e-3, "Jacobian vs finite differences", 0);  // ~1e-6 px absolute

	// LDL^T solve = cv::solve
	const auto residuals = computeReprojectionResiduals(observations, T_cw, K);
	const PoseSystem system = accumulatePoseSystem(residuals, J, observations);
	cv::Vec6d expected;
	cv::Matx66d damped = system.H;
	for (int i = 0; i < 6; ++i) damped(i, i) *= 1.5;
	cv::solve(damped, system.b, expected, cv::DECOMP_CHOLESKY);
	const auto delta = solvePoseIncrement(system.H, system.b, 0.5);
	checkSmall(delta ? cv::norm(*delta - expected) / cv::norm(expected) : 1.0, "LDL^T vs cv::solve", 0);
	checkSmall(solvePoseIncrement(cv::Matx66d::zeros(), system.b).has_value(), "singular -> nullopt", 0);

	// refinePose from a perturbed start converges to the truth within noise
	const cv::Affine3d start = composePose(expSE3(cv::Vec6d(0.05, 0.03, -0.05, 0.02, -0.03, 0.01)), T_cw);
	const auto refined = refinePose(observations, start, K);
	checkSmall(refined ? std::max(0.0, distance(refined->T_cw, T_cw) - 0.01) : 1.0, "refinePose pose", 0);
	checkSmall(refined ? std::max(0.0, refined->finalRms - 0.8) : 1.0, "refinePose rms ~ 0.71", 0);
	checkSmall(refined && refined->initialRms > 10 && refined->numInliers > 280 ? 0 : 1, "refinePose rms drop, inliers", 0);


	// refinePose must not "improve" by pushing points behind the camera (accept only if numValid is unchanged).
	// Start 10 cm behind the truth; 10 weak extra points sit 3-6 cm in front of the start camera.
	// The good points pull the camera forward 10 cm in one step, which would put those 10 points behind it.
	{
		const cv::Affine3d back = composePose(expSE3(cv::Vec6d(0, 0, 0.1, 0, 0, 0)), T_cw);  // camera moved -z
		std::vector<MapObservation> mixed(observations.begin(), observations.begin() + 100);
		for (int i = 0; i < 10; ++i) {
			const cv::Vec3d pc(rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), rng.uniform(0.03, 0.06));
			const cv::Vec3d P = inversePose(back) * pc;
			mixed.push_back({{P[0], P[1], P[2]}, *project({pc[0], pc[1], pc[2]}, K), 1e-6});  // weak: do not resist
		}
		const auto r = refinePose(mixed, back, K);
		size_t behind = 0;
		if (r)
			for (const auto& e : computeReprojectionResiduals(mixed, r->T_cw, K)) behind += std::isnan(e.x);
		checkSmall(r ? double(behind) : 1.0, "refinePose keeps every point in front", 0);
	}

	// buildMapObservations: observed pixel = coords * 1.3^level, not rounded; information = 1.3^(-2 level)
	{
		Frame frame;
		frame.features.push_back({Candidate{0.f, {7, 10}, 4}, {}, 0.f});
		const std::vector<cv::Point3d> mapPoints{{1, 2, 3}};
		const auto obs = buildMapObservations(frame, mapPoints, {{0, 0}});
		const double scale = std::pow(1.3, 4);  // 2.8561: 10 -> 28.561, rounding would give 29
		checkSmall(obs.size() == 1 ? cv::norm(obs[0].pixel - cv::Point2d(7 * scale, 10 * scale)) : 1.0,
		           "map observation pixel unrounded", 4);
		checkSmall(obs.size() == 1 ? std::abs(obs[0].information - 1 / (scale * scale)) : 1.0,
		           "map observation information", 4);
	}

	// Tracking: 200 points seen by a reference camera and the current one, both with depth images.
	Frame reference{{}, cv::Mat::zeros(480, 640, CV_16UC1)}, current{{}, cv::Mat::zeros(480, 640, CV_16UC1)};
	const cv::Affine3d T_rw = expSE3(cv::Vec6d(0.3, 0.1, -0.2, 0.1, 0.2, 0.05));
	const cv::Affine3d T_cr = expSE3(cv::Vec6d(0.05, -0.02, 0.03, 0.02, -0.04, 0.01));
	const cv::Affine3d T_wr = inversePose(T_rw);
	orb_matches_t trackMatches, frameToMap;
	std::vector<cv::Point3d> mapPointsW;
	auto addFeature = [](Frame& f, const cv::Vec3d& pc, const cv::Matx33d& K) {
		const cv::Point2d uv = *project({pc[0], pc[1], pc[2]}, K);
		const cv::Point px(cvRound(uv.x), cvRound(uv.y));
		f.depth.at<uint16_t>(px) = static_cast<uint16_t>(std::lround(pc[2] * DEPTH_SCALE));
		f.features.push_back({{0.0f, {px.x, px.y}, 0}, {}, 0.0f});
	};
	while (trackMatches.size() < 200) {
		const cv::Vec3d pr(rng.uniform(-1.5, 1.5), rng.uniform(-1.0, 1.0), rng.uniform(1.5, 4.0));
		const cv::Vec3d pc = T_cr * pr;
		const auto uvR = project({pr[0], pr[1], pr[2]}, K), uvC = project({pc[0], pc[1], pc[2]}, K);
		if (!uvC || uvC->x < 1 || uvC->y < 1 || uvC->x > 638 || uvC->y > 478 || uvR->x < 1 || uvR->y < 1 || uvR->x > 638 || uvR->y > 478) continue;
		trackMatches.emplace(reference.features.size(), current.features.size());
		frameToMap.emplace(current.features.size(), mapPointsW.size());
		const cv::Vec3d pw = T_wr * pr;
		mapPointsW.emplace_back(pw[0], pw[1], pw[2]);
		addFeature(reference, pr, K);
		addFeature(current, pc, K);
	}
	const cv::Affine3d T_cw_true = composePose(T_cr, T_rw);

	const TrackResult tracked = trackFrame(current, reference, T_rw, trackMatches, mapPointsW, frameToMap, K);
	checkSmall(tracked.status == TrackingStatus::OK ? std::max(0.0, distance(tracked.T_cw, T_cw_true) - 0.005) : 1.0, "trackFrame pose", 0);
	checkSmall(tracked.inlierMatches.size() >= 190 ? 0 : 1, "trackFrame inliers", 0);
	checkSmall(trackFrame(current, reference, T_rw, {}, mapPointsW, frameToMap, K).status == TrackingStatus::LOST ? 0 : 1, "no matches -> LOST", 0);

	const std::vector<RelocalizationCandidate> candidates{{current, T_rw, {}}, {reference, T_rw, trackMatches}};
	const TrackResult relocalized = relocalizeFrame(current, candidates, K);
	checkSmall(relocalized.status == TrackingStatus::OK ? std::max(0.0, distance(relocalized.T_cw, T_cw_true) - 0.01) : 1.0, "relocalizeFrame pose", 0);
	checkSmall(relocalizeFrame(current, {}, K).status == TrackingStatus::LOST ? 0 : 1, "no candidates -> LOST", 0);

	std::printf("%d failures\n", failures);
	return failures != 0;
}
