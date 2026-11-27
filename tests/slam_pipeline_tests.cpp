// End-to-end tests use actual image extraction, descriptor matching and RGB-D tracking.
#include "rgbd_pipeline.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <opencv2/imgproc.hpp>

namespace {

void check(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

// A textured plane at Z=2m. Integer horizontal shifts have the exact known pose
// T_cw.translation.x = shift * 2 / fx, without rendering or pose-estimation oracles.
struct PlaneSequence {
    static constexpr int width = 384;
    static constexpr int height = 256;
    cv::Mat texture;
    cv::Mat depth;

    PlaneSequence()
        : texture(height, width + 64, CV_8UC1), depth(height, width, CV_16UC1, cv::Scalar(10000)) {
        cv::RNG random(0x5A17);
        random.fill(texture, cv::RNG::UNIFORM, 0, 256);
        cv::GaussianBlur(texture, texture, cv::Size(3, 3), 0.6);
    }

    cv::Mat image(int shift) const { return texture(cv::Rect(32 - shift, 0, width, height)).clone(); }
};

slam::PipelineConfig configuration(slam::ExecutionMode mode) {
    slam::PipelineConfig config;
    config.camera.width = PlaneSequence::width;
    config.camera.height = PlaneSequence::height;
    config.camera.fx = config.camera.fy = 300.0;
    config.camera.cx = (PlaneSequence::width - 1) / 2.0;
    config.camera.cy = (PlaneSequence::height - 1) / 2.0;
    config.mode = mode;
    config.maxFeatures = 700;
    config.mapping.minInitialPoints = 60;
    config.mapping.minTrackedPoints = 20;
    config.mapping.referenceTrackedRatio = 1.1; // Exercise mapping even at high overlap.
    config.mapping.maxFramesBetweenKeyframes = 1;
    config.mapping.redundantRatio = 1.1; // Keep this tiny test's observation history.
    config.mapping.baFirstIterations = 3;
    config.mapping.baSecondIterations = 4;
    return config;
}

void checkAssociations(const slam::Map &map) {
    std::size_t repeated = 0;
    for (const auto &[id, landmark] : map.landmarks) {
        if (landmark.bad) continue;
        repeated += landmark.observations.size() > 1;
        for (const auto &[keyframeId, index] : landmark.observations) {
            const auto &keyframe = map.keyframes.at(keyframeId);
            check(!keyframe.bad, "landmark retains an observation in a culled keyframe");
            check(slam::resolveLandmarkId(map, keyframe.landmarkIds.at(static_cast<std::size_t>(index))) ==
                      id,
                  "landmark/keyframe observation mismatch");
        }
    }
    check(repeated >= 30, "repeated image features were not associated with existing landmarks");
    for (const auto &[id, keyframe] : map.keyframes) {
        if (keyframe.bad) continue;
        for (std::size_t index = 0; index < keyframe.landmarkIds.size(); ++index) {
            const int landmarkId = slam::resolveLandmarkId(map, keyframe.landmarkIds[index]);
            if (landmarkId < 0) continue;
            const auto &observations = map.landmarks.at(landmarkId).observations;
            check(observations.count(id) && observations.at(id) == static_cast<int>(index),
                  "keyframe/landmark observation mismatch");
        }
    }
}

std::size_t trajectoryRows(const std::filesystem::path &path) {
    std::ifstream stream(path);
    std::string line;
    std::size_t count = 0;
    double previous = -1.0;
    while (std::getline(stream, line)) {
        std::array<double, 8> values{};
        std::istringstream row(line);
        for (double &value : values)
            check(static_cast<bool>(row >> value) && std::isfinite(value),
                  "invalid exported TUM trajectory row");
        std::string extra;
        check(!(row >> extra), "unexpected extra trajectory columns");
        check(values[0] > previous, "trajectory timestamps are not increasing");
        previous = values[0];
        double norm = 0.0;
        for (std::size_t i = 4; i < values.size(); ++i)
            norm += values[i] * values[i];
        check(std::abs(norm - 1.0) < 1e-5, "exported quaternion is not normalized");
        ++count;
    }
    return count;
}

void testExports(const slam::Map &map) {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir =
        std::filesystem::temp_directory_path() / ("slam-pipeline-test-" + std::to_string(suffix));
    std::filesystem::create_directories(dir);
    try {
        check(slam::exportMap(map, (dir / "map.ply").string()), "map export failed");
        check(slam::exportTrajectory(map, (dir / "trajectory.txt").string()), "trajectory export failed");
        check(slam::exportKeyframeTrajectory(map, (dir / "keyframes.txt").string()),
              "keyframe export failed");
        check(trajectoryRows(dir / "trajectory.txt") == map.frameRecords.size(),
              "export lost tracked frames");
        const auto active = static_cast<std::size_t>(std::count_if(
            map.keyframes.begin(), map.keyframes.end(), [](const auto &entry) { return !entry.second.bad; }));
        check(trajectoryRows(dir / "keyframes.txt") == active, "wrong keyframe trajectory size");
        std::ifstream cloud(dir / "map.ply");
        std::string line;
        std::getline(cloud, line);
        check(line == "ply", "map export is not PLY");
        check(!slam::exportMap(map, (dir / "missing" / "map.ply").string()), "export ignored write failure");
    } catch (...) {
        std::filesystem::remove_all(dir);
        throw;
    }
    std::filesystem::remove_all(dir);
}

slam::Pose testPipeline(const PlaneSequence &sequence, slam::ExecutionMode mode) {
    const auto config = configuration(mode);
    slam::RgbdPipeline pipeline(config);
    std::size_t tracked = 0;
    slam::Pose last;
    const std::array<int, 5> shifts{0, 4, 8, 4, 0};
    for (std::size_t frame = 0; frame < shifts.size(); ++frame) {
        const auto result = pipeline.process(sequence.image(shifts[frame]), sequence.depth, frame / 30.0);
        check(result.features >= 60, "generated texture did not produce enough real features");
        check(result.initialized || result.tracked, "tracking lost on the generated RGB-D plane");
        if (frame == 0)
            check(result.initialized, "first RGB-D image did not initialize the map");
        else {
            check(result.inliers >= 15, "tracking lacks verified RANSAC inliers");
            const double expected = shifts[frame] * 2.0 / config.camera.fx;
            check(std::abs(result.T_cw.t[0] - expected) < 0.02, "estimated horizontal motion is incorrect");
            check(std::abs(result.T_cw.t[1]) < 0.02 && std::abs(result.T_cw.t[2]) < 0.02,
                  "estimated motion drifted away from the plane translation");
            if (shifts[frame] == 8) check(result.T_cw.t[0] > 0.025, "tracking ignored nonzero motion");
        }
        ++tracked;
        last = result.T_cw;
    }
    check(pipeline.map().keyframes.size() >= 2, "keyframe insertion was never exercised");
    checkAssociations(pipeline.map());
    const std::size_t recordsBeforeLost = pipeline.map().frameRecords.size();
    const cv::Mat blank(PlaneSequence::height, PlaneSequence::width, CV_8UC1, cv::Scalar(0));
    const auto lost = pipeline.process(blank, sequence.depth, 5.0 / 30.0);
    check(!lost.tracked && !lost.initialized, "featureless image was accepted as tracked");
    check(pipeline.map().frameRecords.size() == recordsBeforeLost, "LOST frame added a false pose");
    const auto recovered = pipeline.process(sequence.image(0), sequence.depth, 6.0 / 30.0);
    check(recovered.tracked, "known view did not recover tracking after a featureless frame");
    check(recovered.relocalized, "recovery did not report relocalization");
    ++tracked;
    check(pipeline.map().frameRecords.size() == tracked, "tracked trajectory records are incomplete");
    checkAssociations(pipeline.map());
    testExports(pipeline.map());
    std::cout << "pipeline mode=" << static_cast<int>(mode) << ": " << tracked << " tracked frames, "
              << pipeline.map().keyframes.size() << " keyframes\n";
    return last;
}

// Exercise the public defaults with the real extractor: its grid selection may
// yield substantially fewer features than maxFeatures, even on a textured view.
void testDefaultPipeline(const PlaneSequence &sequence) {
    slam::PipelineConfig config;
    config.camera.width = PlaneSequence::width;
    config.camera.height = PlaneSequence::height;
    config.camera.fx = config.camera.fy = 300.0;
    config.camera.cx = (PlaneSequence::width - 1) / 2.0;
    config.camera.cy = (PlaneSequence::height - 1) / 2.0;
    slam::RgbdPipeline pipeline(config);
    slam::PipelineResult last;
    constexpr int frames = 25;
    for (int frame = 0; frame < frames; ++frame) {
        last = pipeline.process(sequence.image(frame), sequence.depth, frame / 30.0);
        if (frame == 0)
            check(last.initialized, "default pipeline failed to initialize a textured RGB-D view");
        check(last.tracked, "default pipeline lost a modest translated RGB-D view");
        const double expected = frame * 2.0 / config.camera.fx;
        check(std::abs(last.T_cw.t[0] - expected) < 0.03, "default pipeline recovered incorrect motion");
    }
    check(last.T_cw.t[0] > 0.1, "default pipeline did not recover the sequence's displacement");
    check(pipeline.map().nextKeyframeId >= 2,
          "default pipeline did not extend the map with another keyframe");
    check(pipeline.map().frameRecords.size() == frames, "default pipeline omitted a tracked frame");
    checkAssociations(pipeline.map());
    std::cout << "default pipeline: " << frames << " tracked frames, " << pipeline.map().nextKeyframeId
              << " keyframes inserted, final x=" << last.T_cw.t[0] << '\n';
}

void testRejectedInput(const PlaneSequence &sequence) {
    auto invalidConfig = configuration(slam::ExecutionMode::Serial);
    invalidConfig.mapping.levels = -1;
    bool rejectedConfig = false;
    try {
        const slam::RgbdPipeline invalid(invalidConfig);
    } catch (const std::invalid_argument &) {
        rejectedConfig = true;
    }
    check(rejectedConfig, "negative pyramid levels were not rejected before allocation");
    invalidConfig = configuration(slam::ExecutionMode::Serial);
    invalidConfig.camera.fx = 0.0;
    rejectedConfig = false;
    try {
        const slam::RgbdPipeline invalid(invalidConfig);
    } catch (const std::invalid_argument &) {
        rejectedConfig = true;
    }
    check(rejectedConfig, "zero focal length was accepted");
    slam::RgbdPipeline pipeline(configuration(slam::ExecutionMode::Serial));
    bool rejected = false;
    try {
        pipeline.process(sequence.image(0), cv::Mat(8, 8, CV_16UC1), 0.0);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "mismatched depth dimensions were accepted");
    rejected = false;
    try {
        pipeline.process(sequence.image(0), cv::Mat(PlaneSequence::height, PlaneSequence::width, CV_32FC1),
                         0.0);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected, "unsupported depth representation was accepted");
    const cv::Mat noDepth(PlaneSequence::height, PlaneSequence::width, CV_16UC1, cv::Scalar(0));
    const auto result = pipeline.process(sequence.image(0), noDepth, 0.0);
    check(!result.initialized && pipeline.map().landmarks.empty(),
          "missing depth initialized an invalid map");
    const auto valid = pipeline.process(sequence.image(0), sequence.depth, 1.0 / 30.0);
    check(valid.initialized, "valid frame could not initialize after rejected inputs");
    rejected = false;
    try {
        pipeline.process(sequence.image(0), sequence.depth, 1.0 / 30.0);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    check(rejected && pipeline.map().frameRecords.size() == 1, "repeated timestamp mutated the trajectory");
}

// The revisit has its own duplicate points and a deliberately drifted pose. Geometry is
// recovered from the actual descriptors/depth, then correction must fuse points and add
// an independently verified loop constraint while keeping the origin fixed.
void testLoopCandidateConsistency(const PlaneSequence &sequence) {
    slam::RgbdPipeline pipeline(configuration(slam::ExecutionMode::Serial));
    check(pipeline.process(sequence.image(0), sequence.depth, 0.0).initialized,
          "loop search fixture initialization");
    for (const auto mode :
         {slam::ExecutionMode::Serial, slam::ExecutionMode::SerialSimd, slam::ExecutionMode::Parallel}) {
        slam::Map map = pipeline.map();
        map.config.keyframesBetweenLoops = 0;
        map.config.loopMinKeyframeGap = 2;
        map.config.loopConsistency = 1;
        const int origin = map.originKeyframeId;
        slam::Frame view = map.keyframes.at(origin).frame;
        view.id = 20;
        slam::Pose drift;
        drift.t[0] = 0.12;
        const int neighbor = slam::insertKeyframe(map, view, drift);
        slam::createLandmarks(map, neighbor, slam::unmatchedFeatures(map, neighbor));
        // This older side of the trajectory is connected by odometry, but its duplicate
        // landmarks have not yet been fused with the revisited origin's landmarks.
        map.keyframes.at(neighbor).parentId = origin;
        map.keyframes.at(origin).childIds.insert(neighbor);
        const auto duplicateIds = map.keyframes.at(neighbor).landmarkIds;
        for (int visit = 0; visit < 2; ++visit) {
            ++view.id;
            const int current = slam::insertKeyframe(map, view, drift);
            for (std::size_t index = 0; index < duplicateIds.size(); ++index)
                if (duplicateIds[index] >= 0)
                    slam::addObservation(map, duplicateIds[index], current, static_cast<int>(index));
            slam::updateCovisibility(map, current);
            const auto result = slam::runLoopClosing(map, current, mode);
            if (visit == 0)
                check(!result.closed, "loop closed before temporal consistency was established");
            else {
                check(result.closed && result.loopKeyframeId == origin && result.loopEdgesInGraph > 0,
                      "consistent distant candidate did not complete loop closing");
                checkAssociations(map);
            }
        }
    }
}

void testLoopCorrection(const PlaneSequence &sequence) {
    slam::RgbdPipeline pipeline(configuration(slam::ExecutionMode::Serial));
    check(pipeline.process(sequence.image(0), sequence.depth, 0.0).initialized,
          "loop fixture initialization");
    slam::Map map = pipeline.map();
    const int origin = map.originKeyframeId;
    slam::Frame revisit = map.keyframes.at(origin).frame;
    revisit.id = 30;
    revisit.timestamp = 1.0;
    slam::Pose drift;
    drift.t[0] = 0.12;
    const int current = slam::insertKeyframe(map, revisit, drift);
    slam::createLandmarks(map, current, slam::unmatchedFeatures(map, current));
    const auto geometry = slam::computeLoopGeometry(map, current, origin, slam::ExecutionMode::Serial);
    check(slam::confirmLoopCandidate(map, current, geometry), "real descriptor/depth revisit did not verify");
    auto badGeometry = geometry;
    badGeometry.geometry.meanError = 1.0;
    check(!slam::confirmLoopCandidate(map, current, badGeometry), "poor loop geometry was accepted");
    const auto corrected = slam::correctLoop(map, current, geometry, slam::ExecutionMode::Serial);
    check(corrected.closed && corrected.loopEdgesInGraph > 0, "verified loop missing from pose graph");
    check(corrected.fusedLandmarks > 30, "loop duplicates were not fused");
    check(corrected.graph.finalChi2 <= corrected.graph.initialChi2 + 1e-10, "pose graph cost increased");
    check(slam::norm(map.keyframes.at(current).T_cw.t) < 0.11, "loop did not reduce injected drift");
    check(slam::norm(map.keyframes.at(origin).T_cw.t) < 1e-12, "loop moved fixed world origin");
    checkAssociations(map);

    // A point must preserve its coordinates in its reference camera during map correction.
    auto &landmark = map.landmarks.begin()->second;
    const int reference = landmark.referenceKeyframeId;
    const slam::Pose oldPose = map.keyframes.at(reference).T_cw;
    slam::Pose newPose = oldPose;
    newPose.t[1] += 0.2;
    const slam::Vec3 cameraPoint = slam::transformPoint(oldPose, landmark.position);
    map.keyframes.at(reference).T_cw = newPose;
    slam::applyMapCorrection(map, {{reference, oldPose}}, {{reference, newPose}}, -2);
    check(slam::norm(slam::transformPoint(newPose, landmark.position) - cameraPoint) < 1e-10,
          "map correction changed reference-camera point coordinates");

    slam::Map unsupported = pipeline.map();
    const auto culled = slam::cullLandmarks(unsupported, 3);
    check(culled > 0, "unsupported landmarks were not culled");
    for (const auto &[id, item] : unsupported.landmarks) {
        (void)id;
        check(item.bad && item.observations.empty(), "culled landmark retained observations");
    }
    std::cout << "loop correction: " << corrected.fusedLandmarks << " duplicate points fused, chi2 "
              << corrected.graph.initialChi2 << " -> " << corrected.graph.finalChi2 << '\n';
}

} // namespace

int main() {
    try {
        const PlaneSequence sequence;
        testRejectedInput(sequence);
        testDefaultPipeline(sequence);
        const auto serial = testPipeline(sequence, slam::ExecutionMode::Serial);
        const auto simd = testPipeline(sequence, slam::ExecutionMode::SerialSimd);
        const auto parallel = testPipeline(sequence, slam::ExecutionMode::Parallel);
        check(slam::norm(serial.t - simd.t) < 1e-5, "SIMD changed the recovered trajectory");
        check(slam::norm(serial.t - parallel.t) < 1e-5, "parallel mode changed the recovered trajectory");
        testLoopCandidateConsistency(sequence);
        testLoopCorrection(sequence);
        std::cout << "RGB-D pipeline tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
