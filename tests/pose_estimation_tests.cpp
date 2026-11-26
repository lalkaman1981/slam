// SE(3) tests; cv::Affine3d is the oracle for compose, inverse and the rotation part of exp.
#include "pose_estimation.hpp"

#include <cstdio>
#include <numbers>

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

	std::printf("%d failures\n", failures);
	return failures != 0;
}
