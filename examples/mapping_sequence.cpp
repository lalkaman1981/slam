// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

// Mapping-only regression/benchmark with synthetic observations and drifting poses.
// Feature matching and loop geometry reuse the teammates' implementations.
// For the actual image-to-map pipeline, run slam_rgbd instead.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "map_pipeline.hpp"

namespace {

using namespace slam;

constexpr double kPi = 3.14159265358979323846;
constexpr int kFramesPerLoop = 300;
constexpr double kCircleRadius = 3.0;
constexpr double kRoomHalfSize = 6.0;
constexpr double kRoomHalfHeight = 1.5;
constexpr int kWallPoints = 9000;
constexpr int kFloorPoints = 3000;
constexpr double kMaxSensorDepth = 7.0;

struct Scene {
    std::vector<Vec3> points;
    std::vector<Descriptor> descriptors;
    std::vector<double> nominalDistance; // distance at which the feature is detected at octave 0
    std::vector<Pose> groundTruth;       // T_cw per frame
};

struct RecordedFrame {
    Frame frame;
    std::vector<int> scenePointIds; // ground-truth point per keypoint, -1 for clutter
};

Pose lookAlong(const Vec3 &center, const Vec3 &forward) {
    const Vec3 z = (1.0 / norm(forward)) * forward;
    Vec3 x = cross(makeVec3(0.0, 1.0, 0.0), z);
    x *= 1.0 / norm(x);
    const Vec3 y = cross(z, x);
    Mat3 Rwc;
    setBlock(Rwc, 0, 0, x);
    setBlock(Rwc, 0, 1, y);
    setBlock(Rwc, 0, 2, z);
    Pose T_wc;
    T_wc.R = Rwc;
    T_wc.t = center;
    return inversePose(T_wc);
}

Descriptor randomDescriptor(std::mt19937 &rng) {
    std::uniform_int_distribution<int> byte(0, 255);
    Descriptor d{};
    for (auto &b : d)
        b = static_cast<std::uint8_t>(byte(rng));
    return d;
}

Scene makeScene(std::mt19937 &rng, int frames) {
    Scene scene;
    std::uniform_real_distribution<double> side(-kRoomHalfSize, kRoomHalfSize);
    std::uniform_real_distribution<double> height(-kRoomHalfHeight, kRoomHalfHeight);
    std::uniform_real_distribution<double> nominal(4.0, 10.0);
    std::uniform_int_distribution<int> wall(0, 3);
    for (int i = 0; i < kWallPoints + kFloorPoints; ++i) {
        Vec3 p;
        if (i < kWallPoints) {
            const double s = side(rng);
            switch (wall(rng)) {
            case 0:
                p = makeVec3(kRoomHalfSize, height(rng), s);
                break;
            case 1:
                p = makeVec3(-kRoomHalfSize, height(rng), s);
                break;
            case 2:
                p = makeVec3(s, height(rng), kRoomHalfSize);
                break;
            default:
                p = makeVec3(s, height(rng), -kRoomHalfSize);
                break;
            }
        } else {
            p = makeVec3(side(rng), (i % 2 == 0) ? kRoomHalfHeight : -kRoomHalfHeight, side(rng));
        }
        scene.points.push_back(p);
        scene.descriptors.push_back(randomDescriptor(rng));
        scene.nominalDistance.push_back(nominal(rng));
    }
    for (int k = 0; k < frames; ++k) {
        const double theta = 2.0 * kPi * k / kFramesPerLoop;
        const Vec3 center = makeVec3(kCircleRadius * std::cos(theta), 0.0, kCircleRadius * std::sin(theta));
        const Vec3 forward = makeVec3(-std::sin(theta), 0.0, std::cos(theta));
        scene.groundTruth.push_back(lookAlong(center, forward));
    }
    // Express everything in the first camera frame, which is the SLAM world origin.
    const Pose T_c0w = scene.groundTruth.front();
    const Pose T_wc0 = inversePose(T_c0w);
    for (Vec3 &p : scene.points)
        p = transformPoint(T_c0w, p);
    for (Pose &T : scene.groundTruth)
        T = composePose(T, T_wc0);
    return scene;
}

RecordedFrame recordFrame(const Scene &scene, int k, const CameraIntrinsics &camera, const Map &map,
                          std::mt19937 &rng) {
    RecordedFrame rec;
    rec.frame.id = k;
    rec.frame.timestamp = k / 30.0;
    std::normal_distribution<double> unit(0.0, 1.0);
    std::uniform_int_distribution<int> bit(0, 255);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const Pose &T = scene.groundTruth[static_cast<std::size_t>(k)];

    auto addKeypoint = [&](double u, double v, int octave, const Descriptor &d, float depth, int sceneId) {
        Keypoint kp;
        kp.u = static_cast<float>(u);
        kp.v = static_cast<float>(v);
        kp.octave = octave;
        rec.frame.keypoints.push_back(kp);
        rec.frame.descriptors.push_back(d);
        rec.frame.depths.push_back(depth);
        rec.scenePointIds.push_back(sceneId);
    };

    for (std::size_t i = 0; i < scene.points.size(); ++i) {
        const Vec3 Xc = transformPoint(T, scene.points[i]);
        if (Xc[2] < 0.3) continue;
        const double ratio = std::log(scene.nominalDistance[i] / Xc[2]) / std::log(map.config.scaleFactor);
        const int octave = static_cast<int>(std::lround(ratio));
        if (octave < 0 || octave >= map.config.levels) continue;
        const double sigma = 0.6 * map.scaleFactors[static_cast<std::size_t>(octave)];
        const double u = camera.fx * Xc[0] / Xc[2] + camera.cx + sigma * unit(rng);
        const double v = camera.fy * Xc[1] / Xc[2] + camera.cy + sigma * unit(rng);
        if (u < 0.0 || v < 0.0 || u >= camera.width || v >= camera.height) continue;
        Descriptor d = scene.descriptors[i];
        for (int flip = 0; flip < 5; ++flip) {
            const int b = bit(rng);
            d[static_cast<std::size_t>(b / 8)] ^= static_cast<std::uint8_t>(1u << (b % 8));
        }
        const float depth =
            Xc[2] < kMaxSensorDepth ? static_cast<float>(Xc[2] * (1.0 + 0.003 * unit(rng))) : 0.0f;
        addKeypoint(u, v, octave, d, depth, static_cast<int>(i));
    }
    // Clutter: features that do not correspond to any scene point.
    for (int c = 0; c < 30; ++c) {
        addKeypoint(uniform(rng) * camera.width, uniform(rng) * camera.height, 0, randomDescriptor(rng),
                    static_cast<float>(1.0 + 5.0 * uniform(rng)), -1);
    }
    return rec;
}

// Stand-in for person 2's trackFrame: predicted pose from drifting odometry, then local-map
// matching by projection (selectLocalMap + findVisibleLandmarks from module 3).
TrackingResult simulateTracking(const Map &map, const Frame &frame, const Pose &predicted,
                                const std::vector<int> &previousTracked, ExecutionMode mode) {
    TrackingResult result;
    result.T_cw = predicted;
    result.landmarkIds.assign(frame.keypoints.size(), -1);
    const LocalMap local = selectLocalMap(map, previousTracked);
    result.referenceKeyframeId = local.referenceKeyframeId;
    const std::vector<VisibleLandmark> visible =
        findVisibleLandmarks(map, local, predicted, map.camera, mode);

    constexpr int kCell = 32;
    const int cols = map.camera.width / kCell + 1;
    const int rows = map.camera.height / kCell + 1;
    std::vector<std::vector<int>> grid(static_cast<std::size_t>(cols * rows));
    for (std::size_t i = 0; i < frame.keypoints.size(); ++i) {
        const int cx = static_cast<int>(frame.keypoints[i].u) / kCell;
        const int cy = static_cast<int>(frame.keypoints[i].v) / kCell;
        grid[static_cast<std::size_t>(cy * cols + cx)].push_back(static_cast<int>(i));
    }

    std::vector<int> bestDistance(frame.keypoints.size(), 256);
    for (const VisibleLandmark &vl : visible) {
        result.visibleLandmarkIds.push_back(vl.landmarkId);
        const Descriptor &d = map.landmarks.at(vl.landmarkId).descriptor;
        const double r = 15.0 * map.scaleFactors[static_cast<std::size_t>(vl.predictedLevel)];
        int best = 256, bestIdx = -1;
        for (int cy = std::max(0, static_cast<int>((vl.v - r) / kCell));
             cy <= std::min(rows - 1, static_cast<int>((vl.v + r) / kCell)); ++cy) {
            for (int cx = std::max(0, static_cast<int>((vl.u - r) / kCell));
                 cx <= std::min(cols - 1, static_cast<int>((vl.u + r) / kCell)); ++cx) {
                for (int idx : grid[static_cast<std::size_t>(cy * cols + cx)]) {
                    const Keypoint &kp = frame.keypoints[static_cast<std::size_t>(idx)];
                    if (std::abs(kp.u - vl.u) > r || std::abs(kp.v - vl.v) > r) continue;
                    if (kp.octave < vl.predictedLevel - 1 || kp.octave > vl.predictedLevel + 1) continue;
                    const int dist = hammingDistance(d, frame.descriptors[static_cast<std::size_t>(idx)]);
                    if (dist < best) {
                        best = dist;
                        bestIdx = idx;
                    }
                }
            }
        }
        if (bestIdx < 0 || best > map.config.descriptorThresholdLow) continue;
        const auto slot = static_cast<std::size_t>(bestIdx);
        if (best < bestDistance[slot]) {
            bestDistance[slot] = best;
            result.landmarkIds[slot] = vl.landmarkId;
        }
    }
    const auto matched =
        std::count_if(result.landmarkIds.begin(), result.landmarkIds.end(), [](int id) { return id >= 0; });
    result.ok = matched >= 30;
    return result;
}

double keyframeAte(const Map &map, const Scene &scene) {
    double sum = 0.0;
    int n = 0;
    for (const auto &[id, kf] : map.keyframes) {
        if (kf.bad) continue;
        const Vec3 est = cameraCenter(kf.T_cw);
        const Vec3 gt = cameraCenter(scene.groundTruth[static_cast<std::size_t>(kf.frame.id)]);
        sum += squaredNorm(est - gt);
        ++n;
    }
    return n > 0 ? std::sqrt(sum / n) : 0.0;
}

double landmarkError(const Map &map, const Scene &scene, const std::vector<RecordedFrame> &recorded) {
    double sum = 0.0;
    int n = 0;
    for (const auto &[id, lm] : map.landmarks) {
        if (lm.bad) continue;
        const Keyframe &ref = map.keyframes.at(lm.referenceKeyframeId);
        const int sceneId = recorded[static_cast<std::size_t>(ref.frame.id)]
                                .scenePointIds[static_cast<std::size_t>(lm.observations.at(ref.id))];
        if (sceneId < 0) continue;
        sum += norm(lm.position - scene.points[static_cast<std::size_t>(sceneId)]);
        ++n;
    }
    return n > 0 ? sum / n : 0.0;
}

std::vector<int> allLandmarkIds(const Map &map) {
    std::vector<int> ids;
    for (const auto &[id, lm] : map.landmarks)
        if (!lm.bad) ids.push_back(id);
    return ids;
}

double runSequence(const std::filesystem::path &outputDir, int frames, ExecutionMode mode) {
    std::filesystem::create_directories(outputDir);

    std::mt19937 rng(42);
    const CameraIntrinsics camera;
    Map map = createMap(camera);
    const Scene scene = makeScene(rng, frames);

    std::vector<RecordedFrame> recorded;
    recorded.reserve(static_cast<std::size_t>(frames));
    for (int k = 0; k < frames; ++k)
        recorded.push_back(recordFrame(scene, k, camera, map, rng));

    std::normal_distribution<double> unit(0.0, 1.0);
    Pose estimate;
    std::vector<int> previousTracked;
    double mapTime = 0.0;
    int keyframes = 0, loops = 0;

    for (int k = 0; k < frames; ++k) {
        const Frame &frame = recorded[static_cast<std::size_t>(k)].frame;
        TrackingResult tracking;
        if (k > 0) {
            // Drifting odometry: true relative motion, Gaussian noise and a small yaw bias.
            const Pose gtRelative =
                composePose(scene.groundTruth[static_cast<std::size_t>(k)],
                            inversePose(scene.groundTruth[static_cast<std::size_t>(k - 1)]));
            Vec6 noise;
            for (std::size_t q = 0; q < 3; ++q)
                noise[q] = 0.001 * unit(rng);
            for (std::size_t q = 3; q < 6; ++q)
                noise[q] = 0.003 * unit(rng);
            noise[1] += 0.002;
            estimate = composePose(composePose(expSE3(noise), gtRelative), estimate);
            tracking = simulateTracking(map, frame, estimate, previousTracked, mode);
        }

        const auto start = std::chrono::steady_clock::now();
        const MapUpdateResult update = updateMap(map, frame, tracking, mode);
        mapTime += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        if (update.keyframeInserted) {
            ++keyframes;
            estimate = map.keyframes.at(update.keyframeId).T_cw;
            previousTracked = map.keyframes.at(update.keyframeId).landmarkIds;
        } else {
            previousTracked = tracking.landmarkIds;
        }

        if (update.initialized) {
            std::cout << "frame " << k << ": map initialized with " << map.landmarks.size() << " points\n";
        } else if (update.keyframeInserted && k % 20 < 4) {
            std::cout << "frame " << k << ": KF " << update.keyframeId << " assoc=" << update.associated
                      << " depth=" << update.createdFromDepth << " tri=" << update.triangulated
                      << " fused=" << update.fused << " culledPts=" << update.culledLandmarks
                      << " culledKFs=" << update.culledKeyframes
                      << " BA[kf=" << update.localBA.optimizedKeyframes << "+"
                      << update.localBA.fixedKeyframes << " pts=" << update.localBA.optimizedPoints
                      << " cost " << update.localBA.initialCost << "->" << update.localBA.finalCost
                      << " outliers=" << update.localBA.outliersRemoved << "] ATE=" << keyframeAte(map, scene)
                      << " m\n";
        }
        if (update.loop.closed) {
            ++loops;
            std::cout << "frame " << k << ": LOOP KF " << update.keyframeId << " <-> KF "
                      << update.loop.loopKeyframeId << " matches=" << update.loop.matches
                      << " fused=" << update.loop.fusedLandmarks
                      << " corrected KFs=" << update.loop.correctedKeyframes
                      << " graph edges=" << update.loop.graphEdges << " (loop "
                      << update.loop.loopEdgesInGraph << ") chi2 " << update.loop.graph.initialChi2 << " -> "
                      << update.loop.graph.finalChi2 << ", ATE after=" << keyframeAte(map, scene) << " m\n";
        }
    }

    int activeKeyframes = 0, activePoints = 0;
    for (const auto &[id, kf] : map.keyframes)
        activeKeyframes += kf.bad ? 0 : 1;
    for (const auto &[id, lm] : map.landmarks)
        activePoints += lm.bad ? 0 : 1;
    std::cout << "\nframes: " << frames << ", keyframes inserted: " << keyframes
              << ", kept: " << activeKeyframes << ", loops closed: " << loops << "\n";
    std::cout << "landmarks created: " << map.landmarks.size() << ", kept: " << activePoints << "\n";
    std::cout << "keyframe ATE (RMSE): " << keyframeAte(map, scene)
              << " m, mean landmark error: " << landmarkError(map, scene, recorded) << " m\n";

    const std::vector<int> ids = allLandmarkIds(map);
    const double before = meanReprojectionError(map, ids);
    LocalMap whole;
    whole.landmarkIds = ids;
    const int refined = refineLandmarks(map, whole, mode);
    std::cout << "structure-only refinement: " << refined << " points improved, mean reprojection error "
              << before << " -> " << meanReprojectionError(map, ids) << " px\n";
    std::cout << "updateMap time: " << mapTime << " s total, " << mapTime / frames * 1000.0
              << " ms per frame\n";

    if (!exportMap(map, (outputDir / "map.ply").string()) ||
        !exportTrajectory(map, (outputDir / "trajectory.txt").string()) ||
        !exportKeyframeTrajectory(map, (outputDir / "keyframe_trajectory.txt").string()))
        throw std::runtime_error("Could not export mapping example output");
    std::cout << "wrote map.ply, trajectory.txt, keyframe_trajectory.txt to " << outputDir.string() << "\n";
    return mapTime;
}

} // namespace

int main(int argc, char *argv[]) {
    try {
        std::filesystem::path outputDir = "mapping-output";
        int frames = 360;
        std::string modeName = "serial";
        bool outputSet = false;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help") {
                std::cout << "Usage: slam_mapping_sequence [output-directory] [--frames N] "
                             "[--mode serial|serial_simd|parallel|all]\n"
                             "The synthetic camera completes a loop at frame 300.\n";
                return 0;
            }
            if (argument == "--frames" || argument == "--mode") {
                if (++i >= argc) throw std::invalid_argument("Missing value after " + argument);
                if (argument == "--frames") {
                    std::size_t consumed = 0;
                    frames = std::stoi(argv[i], &consumed);
                    if (consumed != std::string(argv[i]).size() || frames <= 0)
                        throw std::invalid_argument("--frames must be a positive integer");
                } else
                    modeName = argv[i];
            } else if (!argument.starts_with("--") && !outputSet) {
                outputDir = argument;
                outputSet = true;
            } else
                throw std::invalid_argument("Unknown argument: " + argument);
        }
        const std::vector<std::pair<std::string, slam::ExecutionMode>> modes = {
            {"serial", slam::ExecutionMode::Serial},
            {"serial_simd", slam::ExecutionMode::SerialSimd},
            {"parallel", slam::ExecutionMode::Parallel}};
        if (modeName != "all" && std::none_of(modes.begin(), modes.end(),
                                              [&](const auto &item) { return item.first == modeName; }))
            throw std::invalid_argument("Unknown mapping mode: " + modeName);
        double serialTime = 0.0;
        for (const auto &[name, mode] : modes) {
            if (modeName != "all" && name != modeName) continue;
            std::cout << "Mapping mode: " << name << '\n';
            if (mode == slam::ExecutionMode::SerialSimd)
                std::cerr << "serial_simd backend: " << slam::simdBackendName() << '\n';
            const double elapsed =
                runSequence(modeName == "all" ? outputDir / name : outputDir, frames, mode);
            if (name == "serial") serialTime = elapsed;
            if (modeName == "all")
                std::cout << "Mapping speedup vs serial: " << serialTime / elapsed << "x\n";
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
