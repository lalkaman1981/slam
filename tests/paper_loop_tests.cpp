#include "loop_closing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {

void check(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

struct LoopFixture {
    slam::Map map;
    slam::Pose truth;
    int candidate = -1;
    int current = -1;

    LoopFixture() {
        slam::CameraIntrinsics camera;
        camera.fx = 400;
        camera.fy = 410;
        slam::MapConfig config;
        config.minInitialPoints = 20;
        map = slam::createMap(camera, config);
        slam::Vec6 motion;
        motion[0] = 0.025;
        motion[1] = -0.035;
        motion[2] = 0.01;
        motion[3] = 0.12;
        motion[4] = -0.03;
        motion[5] = 0.04;
        truth = slam::expSE3(motion);
        slam::Frame first, second;
        first.id = 0;
        second.id = 30;
        second.timestamp = 1;
        std::mt19937 random(1729);
        for (int i = 0; i < 80; ++i) {
            const auto point =
                slam::makeVec3((i % 10 - 4.5) * 0.22, (i / 10 - 3.5) * 0.17, 2.5 + 0.07 * (i % 7));
            const auto moved = slam::transformPoint(truth, point);
            slam::Descriptor descriptor;
            for (auto &value : descriptor) value = static_cast<std::uint8_t>(random());
            auto add = [&](slam::Frame &frame, const slam::Vec3 &p) {
                double u, v;
                check(slam::project(p, camera, u, v), "fixture point outside image");
                frame.keypoints.push_back({static_cast<float>(u), static_cast<float>(v), i % 3});
                frame.descriptors.push_back(descriptor);
                frame.depths.push_back(static_cast<float>(p[2]));
            };
            add(first, point);
            add(second, moved);
        }
        candidate = slam::initializeMap(map, first);
        slam::Pose drifted = truth;
        drifted.t[0] += 0.2;
        current = slam::insertKeyframe(map, second, drifted);
        slam::createLandmarks(map, current, slam::unmatchedFeatures(map, current));
        check(map.landmarks.size() == 160, "fixture failed to create duplicate map points");
        // The geometry is already in the map; neither refinement nor RANSAC may
        // require a sensor depth sample from the stored keyframe images.
        for (auto &[id, keyframe] : map.keyframes)
            std::fill(keyframe.frame.depths.begin(), keyframe.frame.depths.end(), 0.0f);
    }
};

double poseError(const slam::Pose &a, const slam::Pose &b) {
    return slam::norm(slam::logSE3(slam::composePose(a, slam::inversePose(b))));
}

slam::Pose checkRefinement(const LoopFixture &fixture, slam::ExecutionMode mode) {
    auto matches = fixture.map.keyframes.at(fixture.candidate).landmarkIds;
    for (std::size_t i = 0; i < 16; ++i) matches[i] = matches[i + 23];
    slam::Vec6 perturbation;
    perturbation[0] = 0.012;
    perturbation[1] = -0.008;
    perturbation[3] = 0.035;
    perturbation[4] = -0.02;
    auto estimate = slam::composePose(slam::expSE3(perturbation), fixture.truth);
    const int inliers =
        slam::refineLoopTransform(fixture.map, fixture.current, fixture.candidate, matches, estimate, mode);
    check(inliers == 64, "bidirectional refinement did not reject the false correspondences");
    check(std::all_of(matches.begin(), matches.begin() + 16, [](int id) { return id == -1; }),
          "outlier correspondence remained attached to the loop");
    check(poseError(estimate, fixture.truth) < 1e-5, "relative loop transform did not converge to truth");
    return estimate;
}

void checkMapGeometry(const LoopFixture &fixture, slam::ExecutionMode mode) {
    const auto geometry = slam::computeLoopGeometry(fixture.map, fixture.current, fixture.candidate, mode);
    check(geometry.success && slam::confirmLoopCandidate(fixture.map, fixture.current, geometry),
          "map-point loop geometry still depends on image depth");
    check(geometry.inlierMatches == 80, "loop geometry lost exact map correspondences");
    check(poseError(geometry.geometry.T_current_candidate, fixture.truth) < 1e-5,
          "map-point loop geometry has wrong pose convention");
    auto corrected = fixture.map;
    const auto result = slam::correctLoop(corrected, fixture.current, geometry, mode);
    check(result.closed && result.fusedLandmarks >= 80, "nonidentity loop did not fuse its duplicate points");
    std::cout << "loop mode=" << static_cast<int>(mode)
              << " pose error " << poseError(fixture.map.keyframes.at(fixture.current).T_cw, fixture.truth)
              << " -> " << poseError(corrected.keyframes.at(fixture.current).T_cw, fixture.truth)
              << ", graph chi2 " << result.graph.initialChi2 << " -> " << result.graph.finalChi2
              << ", parent " << fixture.map.keyframes.at(fixture.current).parentId << " -> "
              << corrected.keyframes.at(fixture.current).parentId << '\n';
    const double initialError = poseError(fixture.map.keyframes.at(fixture.current).T_cw, fixture.truth);
    const double correctedError = poseError(corrected.keyframes.at(fixture.current).T_cw, fixture.truth);
    // Fusion connects this minimal two-keyframe fixture's spanning tree. The old
    // odometry and new loop each have identity information, so the optimum balances
    // their conflicting translations instead of satisfying the loop exactly.
    check(std::abs(correctedError - 0.5 * initialError) < 1e-4,
          "loop graph did not balance its odometry and loop measurements");
    check(result.graph.finalChi2 < result.graph.initialChi2,
          "nonidentity loop optimization did not decrease graph error");
    check(poseError(corrected.keyframes.at(fixture.candidate).T_cw, slam::Pose{}) < 1e-12,
          "loop correction moved the fixed origin");
    check(std::count_if(corrected.landmarks.begin(), corrected.landmarks.end(),
                       [](const auto &entry) { return !entry.second.bad; }) == 80,
          "loop correction left duplicate map points");
}

void checkReprojectionAcceptance(const LoopFixture &fixture) {
    auto uncertainDepth = fixture.map;
    const auto &current = uncertainDepth.keyframes.at(fixture.current);
    for (std::size_t i = 0; i < current.landmarkIds.size(); i += 2) {
        auto &point = uncertainDepth.landmarks.at(current.landmarkIds[i]).position;
        const auto cameraPoint = 1.08 * slam::transformPoint(current.T_cw, point);
        point = slam::transformPoint(slam::inversePose(current.T_cw), cameraPoint);
    }
    const auto geometry = slam::computeLoopGeometry(uncertainDepth, fixture.current, fixture.candidate,
                                                    slam::ExecutionMode::Serial);
    check(geometry.geometry.meanError > 0.05,
          "depth-uncertainty fixture does not exercise the former metric-error gate");
    check(slam::confirmLoopCandidate(uncertainDepth, fixture.current, geometry),
          "consistent image reprojections were rejected for uncertain map-point depth");
    auto invalidQuality = geometry;
    invalidQuality.meanReprojectionChi2 = std::numeric_limits<double>::quiet_NaN();
    check(!slam::confirmLoopCandidate(uncertainDepth, fixture.current, invalidQuality),
          "non-finite loop reprojection quality was accepted");
    auto inconsistentImage = fixture.map;
    for (auto &keypoint : inconsistentImage.keyframes.at(fixture.current).frame.keypoints)
        keypoint.u += 50.0f;
    const auto rejected = slam::computeLoopGeometry(inconsistentImage, fixture.current, fixture.candidate,
                                                   slam::ExecutionMode::Serial);
    check(!slam::confirmLoopCandidate(inconsistentImage, fixture.current, rejected),
          "inconsistent image correspondences passed loop verification");
}

void checkGraphConstraint(const LoopFixture &fixture) {
    const std::map<int, slam::Pose> corrected{{fixture.current, fixture.truth}};
    const std::map<int, slam::Pose> before{{fixture.current, fixture.map.keyframes.at(fixture.current).T_cw}};
    for (const auto &connections : std::array<std::map<int, std::set<int>>, 2>{
             std::map<int, std::set<int>>{},
             {{fixture.current, {fixture.candidate}}, {fixture.candidate, {fixture.current}}}}) {
        const auto graph = slam::buildEssentialGraph(fixture.map, fixture.current, fixture.candidate, before,
                                                     corrected, connections);
        check(std::count_if(graph.edges.begin(), graph.edges.end(),
                            [](const auto &edge) { return edge.type == slam::GraphEdgeType::Loop; }) == 1,
              "verified loop constraint was omitted or duplicated");
        for (const auto &edge : graph.edges)
            if (edge.type == slam::GraphEdgeType::Loop)
                check(slam::norm(slam::computeGraphResidual(edge, graph.poses)) < 1e-12,
                      "new loop was measured using uncorrected poses");
    }
}

} // namespace

int main() {
    try {
        const LoopFixture fixture;
        slam::Pose serial;
        for (const auto mode :
             {slam::ExecutionMode::Serial, slam::ExecutionMode::SerialSimd, slam::ExecutionMode::Parallel}) {
            const auto result = checkRefinement(fixture, mode);
            if (mode == slam::ExecutionMode::Serial)
                serial = result;
            else
                check(poseError(result, serial) < 1e-9, "loop refinement differs between modes");
            checkMapGeometry(fixture, mode);
        }
        checkGraphConstraint(fixture);
        checkReprojectionAcceptance(fixture);
        std::cout << "Paper loop geometry, outlier rejection, no-depth map points and modes passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
