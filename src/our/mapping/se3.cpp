// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include "se3.hpp"
#include "integration_adapters.hpp"

#include <algorithm>
#include <cmath>

namespace slam {

Vec3 transformPoint(const Pose &T, const Vec3 &p) { return T.R * p + T.t; }

Pose composePose(const Pose &a, const Pose &b) {
    return toMappingPose(::composePose(toTrackingPose(a), toTrackingPose(b)));
}

Pose inversePose(const Pose &T) { return toMappingPose(::inversePose(toTrackingPose(T))); }

Vec3 cameraCenter(const Pose &T_cw) { return inversePose(T_cw).t; }

Mat3 expSO3(const Vec3 &omega) {
    return toMappingPose(::expSE3({0.0, 0.0, 0.0, omega[0], omega[1], omega[2]})).R;
}

Vec3 logSO3(const Mat3 &R) {
    Pose pose;
    pose.R = R;
    const cv::Vec6d xi = ::logSE3(toTrackingPose(pose));
    return makeVec3(xi[3], xi[4], xi[5]);
}

Pose expSE3(const Vec6 &xi) {
    // The mapping Jacobians use (omega, upsilon); person 2 uses (rho, phi).
    return toMappingPose(::expSE3({xi[3], xi[4], xi[5], xi[0], xi[1], xi[2]}));
}

Vec6 logSE3(const Pose &T) {
    const cv::Vec6d upstream = ::logSE3(toTrackingPose(T));
    Vec6 result;
    for (std::size_t i = 0; i < 3; ++i) {
        result[i] = upstream[static_cast<int>(i + 3)];
        result[i + 3] = upstream[static_cast<int>(i)];
    }
    return result;
}

Pose applyLeftIncrement(const Pose &T, const Vec6 &delta) { return composePose(expSE3(delta), T); }

std::array<double, 4> rotationToQuaternion(const Mat3 &R) {
    const double trace = R(0, 0) + R(1, 1) + R(2, 2);
    double w = 0.0, x = 0.0, y = 0.0, z = 0.0;
    if (trace > 0.0) {
        const double s = 0.5 / std::sqrt(trace + 1.0);
        w = 0.25 / s;
        x = (R(2, 1) - R(1, 2)) * s;
        y = (R(0, 2) - R(2, 0)) * s;
        z = (R(1, 0) - R(0, 1)) * s;
    } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
        const double s = 2.0 * std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2));
        w = (R(2, 1) - R(1, 2)) / s;
        x = 0.25 * s;
        y = (R(0, 1) + R(1, 0)) / s;
        z = (R(0, 2) + R(2, 0)) / s;
    } else if (R(1, 1) > R(2, 2)) {
        const double s = 2.0 * std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2));
        w = (R(0, 2) - R(2, 0)) / s;
        x = (R(0, 1) + R(1, 0)) / s;
        y = 0.25 * s;
        z = (R(1, 2) + R(2, 1)) / s;
    } else {
        const double s = 2.0 * std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1));
        w = (R(1, 0) - R(0, 1)) / s;
        x = (R(0, 2) + R(2, 0)) / s;
        y = (R(1, 2) + R(2, 1)) / s;
        z = 0.25 * s;
    }
    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    return {x / n, y / n, z / n, w / n};
}

Mat3 quaternionToRotation(double w, double x, double y, double z) {
    const double n = std::sqrt(w * w + x * x + y * y + z * z);
    w /= n;
    x /= n;
    y /= n;
    z /= n;
    Mat3 R;
    R(0, 0) = 1 - 2 * (y * y + z * z);
    R(0, 1) = 2 * (x * y - w * z);
    R(0, 2) = 2 * (x * z + w * y);
    R(1, 0) = 2 * (x * y + w * z);
    R(1, 1) = 1 - 2 * (x * x + z * z);
    R(1, 2) = 2 * (y * z - w * x);
    R(2, 0) = 2 * (x * z - w * y);
    R(2, 1) = 2 * (y * z + w * x);
    R(2, 2) = 1 - 2 * (x * x + y * y);
    return R;
}

} // namespace slam
