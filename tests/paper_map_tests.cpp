#include "map.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
using namespace slam;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

Frame makeFrame(int id, int count) {
    Frame frame;
    frame.id = id;
    frame.timestamp = id;
    for (int i = 0; i < count; ++i) {
        frame.keypoints.push_back({static_cast<float>(180 + i * 7), 200.0f, 0, 0.0f, 1.0f});
        Descriptor descriptor;
        std::uint32_t state = 12345u + static_cast<std::uint32_t>(i) * 6789u;
        for (auto &byte : descriptor) {
            state = state * 1664525u + 1013904223u;
            byte = static_cast<std::uint8_t>(state >> 24);
        }
        frame.descriptors.push_back(descriptor);
        frame.depths.push_back(2.0f);
    }
    return frame;
}

Map makeMap() {
    MapConfig config;
    config.minInitialPoints = 1;
    return createMap({}, config);
}

void testCovisibility() {
    auto map = makeMap();
    const int origin = initializeMap(map, makeFrame(0, 25));
    const int strong = insertKeyframe(map, makeFrame(1, 25), {});
    const int weak = insertKeyframe(map, makeFrame(2, 25), {});
    for (int i = 0; i < 25; ++i)
        addObservation(map, i, i < 20 ? strong : weak, i);
    updateCovisibility(map, origin);

    require(map.keyframes.at(origin).covisibilityWeights.at(weak) == 5, "weak connection lost");
    require(map.keyframes.at(weak).covisibilityWeights.at(origin) == 5, "missing reciprocal weak link");
    require(connectedKeyframes(map, origin) == std::vector<int>({strong, weak}), "all connected frames");
    require(covisibleKeyframes(map, origin) == std::vector<int>({strong}), "strong-neighbor selection");
    require(bestCovisibleKeyframes(map, weak, 10) == std::vector<int>({origin}), "strongest fallback");
    require(covisibleKeyframes(map, weak, 10).empty(), "explicit threshold must have no fallback");

    updateCovisibility(map, weak);
    require(map.keyframes.at(origin).covisibilityWeights.at(strong) == 20,
            "updating weak neighbor removed unrelated connection");
    for (int i = 20; i < 25; ++i)
        setLandmarkBad(map, i);
    updateCovisibility(map, origin);
    require(!map.keyframes.at(origin).covisibilityWeights.contains(weak) &&
                !map.keyframes.at(weak).covisibilityWeights.contains(origin),
            "removed shared points left a stale reciprocal link");
}

void testActiveKeyframeCount() {
    auto map = makeMap();
    map.config.maxFramesAfterRelocalization = 2;
    map.config.minTrackedPoints = 1;
    map.config.useCloseDepthRule = false;
    const int origin = initializeMap(map, makeFrame(0, 4));
    for (int i = 1; i <= 3; ++i) {
        const int id = insertKeyframe(map, makeFrame(i, 0), {});
        map.keyframes.at(id).parentId = origin;
        map.keyframes.at(origin).childIds.insert(id);
        setKeyframeBad(map, id);
    }
    map.lastRelocalizationFrameId = 9;
    TrackingResult tracking;
    tracking.ok = true;
    tracking.referenceKeyframeId = origin;
    tracking.landmarkIds = {0, 1, -1, -1};
    require(shouldInsertKeyframe(makeFrame(10, 4), tracking, map),
            "culled trajectory references counted as active keyframes");
    map.config.localMappingIdle = false;
    require(!shouldInsertKeyframe(makeFrame(100, 4), tracking, map),
            "busy local mapper accepted a keyframe");
}

void testMeanViewingDirection() {
    auto map = makeMap();
    const int origin = initializeMap(map, makeFrame(0, 1));
    Pose secondPose;
    secondPose.t = makeVec3(-std::sqrt(3.0), 0.0, -1.0);
    const int second = insertKeyframe(map, makeFrame(1, 1), secondPose);
    map.keyframes.at(origin).T_cw.t = makeVec3(std::sqrt(3.0), 0.0, -1.0);
    map.landmarks.at(0).position = makeVec3(0.0, 0.0, 2.0);
    addObservation(map, 0, second, 0);
    updateNormalAndDepth(map, 0);
    require(norm(map.landmarks.at(0).normal - makeVec3(0.0, 0.0, 0.5)) < 1e-12,
            "mean viewing ray was renormalized and lost angular spread");
}

void testFusionDepthGate() {
    auto map = makeMap();
    const auto reference = makeFrame(0, 1);
    initializeMap(map, reference);
    auto target = makeFrame(1, 1);
    target.depths[0] = 1.0f;
    const int id = insertKeyframe(map, target, {});
    require(fuseLandmarksIntoKeyframe(map, id, {0}, 3.0, true, false) == 0,
            "fusion accepted matching pixels at conflicting depths");
    require(map.keyframes.at(id).landmarkIds[0] < 0, "rejected fusion changed associations");
    map.keyframes.at(id).frame.depths[0] = 2.0f;
    require(fuseLandmarksIntoKeyframe(map, id, {0}, 3.0, true, false) == 1,
            "fusion rejected consistent pixels and depth");
    require(map.landmarks.at(0).observations.at(id) == 0, "fusion observation is not reciprocal");
}

void testTriangulationWithoutOrientationHistogram() {
    auto map = makeMap();
    auto reference = makeFrame(0, 6);
    auto current = makeFrame(1, 6);
    std::vector<Vec3> positions;
    Pose pose;
    pose.t = makeVec3(-0.3, 0.0, 0.0);
    for (int i = 0; i < 6; ++i) {
        const float u = 180.0f + 50.0f * i;
        const float v = 170.0f + 15.0f * i;
        positions.push_back(makeVec3((u - map.camera.cx) * 3.0 / map.camera.fx,
                                     (v - map.camera.cy) * 3.0 / map.camera.fy, 3.0));
        reference.keypoints[i] = {u, v, 0, 0.0f, 1.0f};
        current.keypoints[i] = {u - 52.5f, v, 0, 30.0f * i, 1.0f};
        reference.depths[i] = current.depths[i] = i == 5 ? 3.0f : 0.0f;
    }
    initializeMap(map, reference);
    const int currentId = insertKeyframe(map, current, pose);
    addObservation(map, 0, currentId, 5); // One known point supplies the covisibility link.
    updateCovisibility(map, currentId);
    require(triangulateNewLandmarks(map, currentId) == 5,
            "triangulation discarded geometrically valid matches by their orientations");
    for (int i = 0; i < 5; ++i) {
        const int id = map.keyframes.at(currentId).landmarkIds[i];
        require(id >= 0 && norm(map.landmarks.at(id).position - positions[i]) < 1e-5,
                "triangulated point differs from known geometry");
    }
}

void testReplacementChain() {
    auto map = makeMap();
    for (int i = 0; i <= 80; ++i) {
        Landmark landmark;
        landmark.id = i;
        landmark.bad = i != 80;
        landmark.replacedBy = i + 1;
        map.landmarks.emplace(i, landmark);
    }
    require(resolveLandmarkId(map, 0) == 80, "long valid fusion chain was truncated");
    map.landmarks.at(80).bad = true;
    map.landmarks.at(80).replacedBy = 0;
    require(resolveLandmarkId(map, 0) == -1, "replacement cycle was not rejected");
}
} // namespace

int main() {
    try {
        testCovisibility();
        testActiveKeyframeCount();
        testMeanViewingDirection();
        testFusionDepthGate();
        testTriangulationWithoutOrientationHistogram();
        testReplacementChain();
        std::cout << "paper map tests passed\n";
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
