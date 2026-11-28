#include "bundle_adjustment.hpp"
#include "execution.hpp"
#include "pose_graph.hpp"
#include "loop_closing.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
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

// Exercise the linear solver through known solutions, including odd SIMD tails
// and uniformly rescaled systems whose conditioning is unchanged.
void testCholesky() {
    for (const std::size_t size : {1, 2, 3, 7, 16, 31}) {
        for (const double scale : {1e-100, 1.0, 1e100}) {
            DenseMatrix matrix(size, size);
            std::vector<double> truth(size), rhs(size, 0.0), serial;
            for (std::size_t i = 0; i < size; ++i)
                truth[i] = (i % 2 ? -1.0 : 1.0) * static_cast<double>(i + 1) / size;
            for (std::size_t i = 0; i < size; ++i) {
                for (std::size_t j = 0; j < size; ++j) {
                    const auto distance = i > j ? i - j : j - i;
                    matrix(i, j) = scale * ((i == j ? 2.0 * size : 0.0) + 1.0 / (1.0 + distance));
                    rhs[i] += matrix(i, j) * truth[j];
                }
            }
            for (const auto mode : {ExecutionMode::Serial, ExecutionMode::SerialSimd,
                                    ExecutionMode::Parallel}) {
                std::vector<double> solution;
                require(solveCholesky(matrix, rhs, solution, mode), "scaled SPD system rejected");
                if (mode == ExecutionMode::Serial) serial = solution;
                for (std::size_t i = 0; i < size; ++i) {
                    close(solution[i], truth[i], 1e-12, "Cholesky known solution");
                    close(solution[i], serial[i], 1e-12, "Cholesky SIMD parity");
                }
            }
        }
    }
    for (const auto mode : {ExecutionMode::Serial, ExecutionMode::SerialSimd}) {
        std::vector<double> solution;
        require(!solveCholesky(DenseMatrix(2, 3), {1.0, 1.0}, solution, mode),
                "nonsquare Cholesky input accepted");
        require(!solveCholesky(DenseMatrix(2, 2), {1.0}, solution, mode),
                "mismatched Cholesky RHS accepted");
        DenseMatrix invalid(2, 2, 1.0);
        require(!solveCholesky(invalid, {1.0, 1.0}, solution, mode), "singular system accepted");
        invalid(0, 1) = invalid(1, 0) = 2.0;
        require(!solveCholesky(invalid, {1.0, 1.0}, solution, mode), "indefinite system accepted");
        invalid(0, 1) = invalid(1, 0) = 0.0;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        require(!solveCholesky(invalid, {nan, 1.0}, solution, mode), "nonfinite RHS accepted");
        invalid(0, 0) = std::numeric_limits<double>::infinity();
        require(!solveCholesky(invalid, {1.0, 1.0}, solution, mode), "infinite diagonal accepted");
        invalid(0, 0) = 1.0;
        invalid(1, 0) = nan;
        require(!solveCholesky(invalid, {1.0, 1.0}, solution, mode), "nonfinite matrix accepted");
    }
}

Map singleLandmarkMap(const std::vector<int> &octaves) {
    Map map = createMap(CameraIntrinsics{});
    map.originKeyframeId = 0;
    Landmark landmark;
    landmark.id = 0;
    landmark.referenceKeyframeId = 0;
    landmark.position = makeVec3(2.5 * 3.0 / map.camera.fx, 0.0, 3.0);
    for (std::size_t i = 0; i < octaves.size(); ++i) {
        Keyframe keyframe;
        keyframe.id = static_cast<int>(i);
        keyframe.frame.keypoints.push_back(
            {static_cast<float>(map.camera.cx), static_cast<float>(map.camera.cy), octaves[i], 0, 0});
        keyframe.frame.depths.push_back(3.0f);
        keyframe.frame.descriptors.push_back({});
        keyframe.landmarkIds.push_back(0);
        landmark.observations[keyframe.id] = 0;
        map.keyframes.emplace(keyframe.id, std::move(keyframe));
    }
    map.landmarks.emplace(0, std::move(landmark));
    updateNormalAndDepth(map, 0);
    return map;
}

void testLocalBundleAdjustment() {
    for (const auto mode : {ExecutionMode::Serial, ExecutionMode::SerialSimd, ExecutionMode::Parallel}) {
        auto fixed = singleLandmarkMap({0});
        const Pose anchor = fixed.keyframes.at(0).T_cw;
        const auto result = localBundleAdjustment(fixed, 0, mode);
        require(result.optimizedKeyframes == 0 && result.fixedKeyframes == 1,
                "fixed-camera BA fixture acquired a movable camera");
        require(result.finalCost < result.initialCost * 1e-6, "fixed-camera BA skipped point refinement");
        closeMatrix(fixed.keyframes.at(0).T_cw.R, anchor.R, 0.0, "fixed-camera BA moved rotation");
        closeMatrix(fixed.keyframes.at(0).T_cw.t, anchor.t, 0.0, "fixed-camera BA moved translation");
        closeMatrix(fixed.landmarks.at(0).position, makeVec3(0, 0, 3), 1e-6,
                    "fixed-camera BA failed to recover point");

        // The fine-scale observation is initially an outlier; three coarse-scale
        // observations remain inliers and move the point so that it fits all four.
        // ORB-SLAM's first-pass rejection must persist even when its final fit is good.
        auto rejected = singleLandmarkMap({2, 2, 2, 0});
        rejected.config.baFirstIterations = 0;
        rejected.config.baSecondIterations = 10;
        const auto removed = localBundleAdjustment(rejected, 3, mode);
        const auto &landmark = rejected.landmarks.at(0);
        require(removed.outliersRemoved == 1 && !landmark.bad && landmark.observations.size() == 3,
                "first-pass BA outlier was resurrected");
        require(landmark.observations.count(3) == 0 && rejected.keyframes.at(3).landmarkIds[0] == -1,
                "BA outlier removal left inconsistent observations");
        closeMatrix(landmark.position, makeVec3(0, 0, 3), 1e-6, "second-pass BA failed to refine point");
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

template <std::size_t R, std::size_t K, std::size_t C> void checkSimdProduct() {
    Matrix<R, K> a;
    Matrix<K, C> b;
    for (std::size_t i = 0; i < R * K; ++i) a[i] = std::sin(0.37 * static_cast<double>(i) + 1.0) * 3.0;
    for (std::size_t i = 0; i < K * C; ++i) b[i] = std::cos(0.11 * static_cast<double>(i)) * 2.0;
    closeMatrix(detail::multiply(a, b, ExecutionMode::SerialSimd), a * b, 1e-14, "SIMD matrix product");
}

// Backend-independent checks of the serial_simd building blocks (run under every
// SLAM_SIMD_BACKEND): lane order, horizontal sums, every dot-product tail and odd widths.
void testSimdPrimitives() {
    using detail::Double2;
    require(std::string(simdBackendName()).size() > 0, "SIMD backend name");
    const double in[2]{1.5, -7.25};
    double out[2]{};
    Double2::load(in).store(out);
    require(out[0] == 1.5 && out[1] == -7.25, "SIMD load/store lane order");
    Double2(3.0, 4.0).store(out);
    require(out[0] == 3.0 && out[1] == 4.0, "SIMD constructor lane order");
    require(Double2(3.0, 4.0).sum() == 7.0, "SIMD horizontal sum");
    (-(Double2(8.0, 9.0) / Double2(2.0, 3.0) - Double2(1.0))).store(out);
    require(out[0] == -3.0 && out[1] == -2.0, "SIMD arithmetic lanes");
    for (std::size_t n = 0; n <= 19; ++n) {
        std::vector<double> a(n), b(n);
        double reference = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            a[i] = std::sin(0.7 * static_cast<double>(i) + 1.0) * 10.0;
            b[i] = std::cos(1.3 * static_cast<double>(i)) * 0.1;
            reference += a[i] * b[i];
        }
        close(detail::innerProduct(a, b, ExecutionMode::SerialSimd), reference, 1e-14, "SIMD dot-product tail");
    }
    checkSimdProduct<3, 6, 6>();
    checkSimdProduct<6, 3, 3>();
    checkSimdProduct<1, 5, 7>();
    checkSimdProduct<6, 6, 1>();
}
} // namespace
int main() {
    try {
        std::cout << "serial_simd backend: " << simdBackendName() << '\n';
        testSimdPrimitives();
        testResidualsAndJacobians();
        testBundleAdjustment();
        testCholesky();
        testLocalBundleAdjustment();
        testPoseGraph();
        testLandmarksAndCorrection();
        std::cout << "Mapping modes, Jacobians, BA, graph, refinement and correction passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Mapping test failed: " << e.what() << '\n';
        return 1;
    }
}
