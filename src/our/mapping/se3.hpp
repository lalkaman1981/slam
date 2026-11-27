#pragma once

// Shared rigid-body math. The team plan assigns composePose / inversePose / expSE3 / logSE3
// to person 2; adapters below delegate to that implementation:
//   * Pose T_ab maps points from frame b to frame a: x_a = R * x_b + t.
//     Keyframe and frame poses are stored as T_cw (world -> camera), as in ORB-SLAM.
//   * A 6-vector twist is ordered [omega_x, omega_y, omega_z, upsilon_x, upsilon_y, upsilon_z]
//     (rotation first, as g2o's SE3Quat used by ORB-SLAM).
//   * Increments are applied on the left: T <- expSE3(delta) * T.

#include <array>

#include "linalg.hpp"

namespace slam {

struct Pose {
    Mat3 R = Mat3::identity();
    Vec3 t{};
};

Vec3 transformPoint(const Pose &T, const Vec3 &p);
Pose composePose(const Pose &a, const Pose &b); // a * b
Pose inversePose(const Pose &T);
Vec3 cameraCenter(const Pose &T_cw); // camera origin in world coordinates

Mat3 expSO3(const Vec3 &omega);
Vec3 logSO3(const Mat3 &R);
Pose expSE3(const Vec6 &xi);
Vec6 logSE3(const Pose &T);
Pose applyLeftIncrement(const Pose &T, const Vec6 &delta);

// Unit quaternion (x, y, z, w) <-> rotation matrix.
std::array<double, 4> rotationToQuaternion(const Mat3 &R);
Mat3 quaternionToRotation(double w, double x, double y, double z);

} // namespace slam
