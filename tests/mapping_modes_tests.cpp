#include "bundle_adjustment.hpp"
#include "pose_graph.hpp"
#include "loop_closing.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace slam;
void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void close(double a, double b, double tolerance, const char *message) {
    require(std::isfinite(a) && std::isfinite(b) && std::abs(a - b) <= tolerance * (1.0 + std::abs(a)),
            message);
}
template <std::size_t R, std::size_t C>
void closeMatrix(const Matrix<R, C> &a, const Matrix<R, C> &b, double tolerance, const char *message) {
    for (std::size_t i = 0; i < R * C; ++i)
        close(a[i], b[i], tolerance, message);
}

BAProblem makeProblem() {
    BAProblem p;
    p.poses.resize(3);
    p.poses[1].t = makeVec3(-0.2, 0.03, 0.02);
    p.poses[2].t = makeVec3(-0.4, -0.03, 0.01);
    p.fixedPose = {true, false, false};
    for (int j = 0; j < 81; ++j) {
        const Vec3 truth = makeVec3((j % 9 - 4) * 0.2, (j / 9 - 4) * 0.15, 3.0 + 0.1 * (j % 7));
        p.points.push_back(truth + makeVec3(0.008, -0.006, 0.012));
        for (std::size_t i = 0; i < 3; ++i) {
            const auto x = transformPoint(p.poses[i], truth);
            BAObservation obs;
            obs.poseIndex = i;
            obs.pointIndex = static_cast<std::size_t>(j);
            obs.hasDepth = (j % 3) != 0;
            obs.information = 1.0 / (1.0 + static_cast<double>(j % 4));
            const double u = p.camera.fx * x[0] / x[2] + p.camera.cx;
            obs.measured = makeVec3(u, p.camera.fy * x[1] / x[2] + p.camera.cy,
                                    obs.hasDepth ? u - p.camera.bf / x[2] : 0.0);
            p.observations.push_back(obs);
        }
    }
    p.poses[1].t += makeVec3(0.006, -0.009, 0.003);
    p.poses[2].t += makeVec3(-0.005, 0.002, -0.004);
    return p;
}

void testResidualsAndJacobians() {
    const auto p = makeProblem();
    std::vector<ObservationResidual> serial;
    computeObservationResiduals(p, serial, true, ExecutionMode::Serial);
    for (auto mode : {ExecutionMode::SerialSimd, ExecutionMode::Parallel}) {
        std::vector<ObservationResidual> other;
        computeObservationResiduals(p, other, true, mode);
        require(serial.size() == other.size(), "residual size");
        for (std::size_t i = 0; i < serial.size(); ++i) {
            closeMatrix(serial[i].error, other[i].error, 1e-11, "residual parity");
            closeMatrix(serial[i].jacobianPose, other[i].jacobianPose, 1e-11, "pose Jacobian parity");
            closeMatrix(serial[i].jacobianPoint, other[i].jacobianPoint, 1e-11, "point Jacobian parity");
            close(serial[i].chi2, other[i].chi2, 1e-10, "chi2 parity");
            require(serial[i].inFront == other[i].inFront, "front parity");
        }
    }
    const auto &obs = p.observations[4];
    const auto &T = p.poses[obs.poseIndex];
    const auto &X = p.points[obs.pointIndex];
    const auto analytic = evaluateObservation(T, X, obs, p.camera, true);
    constexpr double epsilon = 1e-6;
    for (std::size_t q = 0; q < 6; ++q) {
        Vec6 d;
        d[q] = epsilon;
        const auto plus = evaluateObservation(applyLeftIncrement(T, d), X, obs, p.camera, false);
        const auto minus = evaluateObservation(applyLeftIncrement(T, -d), X, obs, p.camera, false);
        for (std::size_t r = 0; r < 3; ++r)
            close(analytic.jacobianPose(r, q), (plus.error[r] - minus.error[r]) / (2 * epsilon), 1e-6,
                  "pose finite difference");
    }
    for (std::size_t q = 0; q < 3; ++q) {
        Vec3 d;
        d[q] = epsilon;
        const auto plus = evaluateObservation(T, X + d, obs, p.camera, false);
        const auto minus = evaluateObservation(T, X - d, obs, p.camera, false);
        for (std::size_t r = 0; r < 3; ++r)
            close(analytic.jacobianPoint(r, q), (plus.error[r] - minus.error[r]) / (2 * epsilon), 1e-6,
                  "point finite difference");
    }
    // Includes the scalar tail, behind-camera and near-zero-depth cases.
    auto exceptional = p;
    exceptional.points[0][2] = -1.0;
    exceptional.points[1][2] = 0.0;
    std::vector<ObservationResidual> a, b;
    computeObservationResiduals(exceptional, a, true, ExecutionMode::Serial);
    computeObservationResiduals(exceptional, b, true, ExecutionMode::SerialSimd);
    for (std::size_t i = 0; i < a.size(); ++i) {
        closeMatrix(a[i].error, b[i].error, 1e-10, "exceptional projection parity");
        require(a[i].inFront == b[i].inFront, "exceptional front parity");
    }
}

void testBundleAdjustment() {
    auto serial = makeProblem();
    const auto initial = serial;
    const double serialCost = optimizeBAProblem(serial, 12, true, ExecutionMode::Serial);
    require(serialCost < 1e-4, "BA must converge on exact observations");
    closeMatrix(serial.poses[0].t, initial.poses[0].t, 0.0, "BA gauge fixed");
    for (auto mode : {ExecutionMode::SerialSimd, ExecutionMode::Parallel}) {
        auto p = initial;
        const double cost = optimizeBAProblem(p, 12, true, mode);
        require(cost < 1e-4, "variant BA convergence");
        close(cost, serialCost, 1e-7, "BA cost parity");
        for (std::size_t i = 0; i < p.poses.size(); ++i) {
            closeMatrix(p.poses[i].R, serial.poses[i].R, 1e-6, "BA rotation parity");
            closeMatrix(p.poses[i].t, serial.poses[i].t, 1e-6, "BA translation parity");
        }
        for (std::size_t i = 0; i < p.points.size(); ++i)
            closeMatrix(p.points[i], serial.points[i], 1e-6, "BA point parity");
    }
}

PoseGraph makeGraph() {
    PoseGraph g;
    for (int i = 0; i < 12; ++i) {
        Pose pose;
        pose.t = makeVec3(0.11 * i, 0.005 * i, 0.0);
        addGraphNode(g, i, pose, i == 0);
        if (i > 0) {
            Pose delta;
            delta.t = makeVec3(-0.1, 0.0, 0.0);
            addOdometryEdge(g, i - 1, i, delta);
        }
    }
    Pose loop;
    loop.t = makeVec3(-1.1, 0.0, 0.0);
    addLoopEdge(g, 0, 11, loop);
    return g;
}
void testPoseGraph() {
    const auto initial = makeGraph();
    const auto a = linearizePoseGraph(initial, ExecutionMode::Serial);
    for (auto mode : {ExecutionMode::SerialSimd, ExecutionMode::Parallel}) {
        const auto b = linearizePoseGraph(initial, mode);
        close(a.chi2, b.chi2, 1e-12, "graph cost parity");
        for (std::size_t r = 0; r < a.b.size(); ++r) {
            close(a.b[r], b.b[r], 1e-12, "graph gradient parity");
            for (std::size_t c = 0; c < a.b.size(); ++c)
                close(a.H(r, c), b.H(r, c), 1e-12, "graph Hessian parity");
        }
    }
    for (auto mode : {ExecutionMode::Serial, ExecutionMode::SerialSimd, ExecutionMode::Parallel}) {
        auto g = initial;
        const auto result = optimizePoseGraph(g, 20, mode);
        require(result.finalChi2 < result.initialChi2 * 1e-6, "pose graph decreases drift");
        closeMatrix(g.poses[0].t, initial.poses[0].t, 0.0, "graph anchor fixed");
        close(g.poses.back().t[0], 1.1, 1e-6, "loop corrects endpoint");
    }
}

Map makeMap() {
    Map map = createMap(CameraIntrinsics{});
    map.originKeyframeId = 0;
    const auto problem = makeProblem();
    for (std::size_t i = 0; i < problem.poses.size(); ++i) {
        Keyframe kf;
        kf.id = static_cast<int>(i);
        kf.frame.id = kf.id;
        kf.T_cw = problem.poses[i];
        for (const auto &obs : problem.observations) {
            if (obs.poseIndex != i) continue;
            kf.frame.keypoints.push_back(
                {static_cast<float>(obs.measured[0]), static_cast<float>(obs.measured[1]), 0, 0, 0});
            const auto x = transformPoint(problem.poses[i], problem.points[obs.pointIndex]);
            kf.frame.depths.push_back(static_cast<float>(x[2]));
            kf.frame.descriptors.push_back({});
            kf.landmarkIds.push_back(static_cast<int>(obs.pointIndex));
        }
        map.keyframes[kf.id] = kf;
    }
    for (std::size_t j = 0; j < problem.points.size(); ++j) {
        Landmark lm;
        lm.id = static_cast<int>(j);
        lm.position = problem.points[j];
        lm.referenceKeyframeId = 0;
        for (int i = 0; i < 3; ++i)
            lm.observations[i] = static_cast<int>(j);
        map.landmarks[lm.id] = lm;
        updateNormalAndDepth(map, lm.id);
    }
    return map;
}

void testLandmarksAndCorrection() {
    const auto initial = makeMap();
    LocalMap local;
    for (const auto &[id, lm] : initial.landmarks)
        local.landmarkIds.push_back(id);
    const auto a = findVisibleLandmarks(initial, local, Pose{}, initial.camera, ExecutionMode::Serial);
    auto serial = initial;
    const int refined = refineLandmarks(serial, local, ExecutionMode::Serial);
    require(refined > 0, "structure refinement improves points");
    for (auto mode : {ExecutionMode::SerialSimd, ExecutionMode::Parallel}) {
        const auto b = findVisibleLandmarks(initial, local, Pose{}, initial.camera, mode);
        require(a.size() == b.size(), "visible count parity");
        for (std::size_t i = 0; i < a.size(); ++i) {
            require(a[i].landmarkId == b[i].landmarkId, "visible order parity");
            close(a[i].u, b[i].u, 1e-12, "visible projection parity");
        }
        auto map = initial;
        require(refineLandmarks(map, local, mode) == refined, "refinement count parity");
        for (const auto &[id, lm] : map.landmarks)
            closeMatrix(lm.position, serial.landmarks.at(id).position, 1e-8, "refinement parity");
    }
    auto map = initial;
    std::map<int, Pose> oldPoses, newPoses;
    for (auto &[id, kf] : map.keyframes) {
        oldPoses[id] = kf.T_cw;
        kf.T_cw.t[0] += 0.2;
        newPoses[id] = kf.T_cw;
    }
    applyMapCorrection(map, oldPoses, newPoses, 99);
    for (const auto &[id, lm] : map.landmarks) {
        const auto &before = initial.landmarks.at(id);
        closeMatrix(transformPoint(oldPoses.at(0), before.position),
                    transformPoint(newPoses.at(0), lm.position), 1e-12, "map follows corrected reference");
    }
}
} // namespace
int main() {
    try {
        testResidualsAndJacobians();
        testBundleAdjustment();
        testPoseGraph();
        testLandmarksAndCorrection();
        std::cout << "Mapping modes, Jacobians, BA, graph, refinement and correction passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Mapping test failed: " << e.what() << '\n';
        return 1;
    }
}
