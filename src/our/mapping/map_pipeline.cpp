// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include "map_pipeline.hpp"

#include <fstream>
#include <iomanip>
#include <set>

namespace slam {

namespace {

void writeTumLine(std::ofstream &out, double timestamp, const Pose &T_cw) {
    const Pose T_wc = inversePose(T_cw);
    const auto q = rotationToQuaternion(T_wc.R);
    out << std::fixed << std::setprecision(6) << timestamp << std::setprecision(7) << ' ' << T_wc.t[0] << ' '
        << T_wc.t[1] << ' ' << T_wc.t[2] << ' ' << q[0] << ' ' << q[1] << ' ' << q[2] << ' ' << q[3] << '\n';
}

// Updates the visible / found counters used by the recent-point culling test (Sec. VI-B).
void updateTrackingCounters(Map &map, const TrackingResult &tracking) {
    std::set<int> visible, found;
    for (int slot : tracking.visibleLandmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id >= 0) visible.insert(id);
    }
    for (int slot : tracking.landmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id < 0) continue;
        visible.insert(id);
        found.insert(id);
    }
    for (int id : visible)
        ++map.landmarks.at(id).visibleCount;
    for (int id : found)
        ++map.landmarks.at(id).foundCount;
}

} // namespace

MapUpdateResult updateMap(Map &map, const Frame &frame, const TrackingResult &tracking, ExecutionMode mode) {
    MapUpdateResult result;
    if (map.keyframes.empty()) {
        result.keyframeId = initializeMap(map, frame);
        result.initialized = result.keyframeId >= 0;
        result.keyframeInserted = result.initialized;
        return result;
    }
    if (!tracking.ok) return result;
    if (tracking.relocalized) map.lastRelocalizationFrameId = frame.id;
    updateTrackingCounters(map, tracking);

    int referenceId = tracking.referenceKeyframeId;
    if (map.keyframes.count(referenceId) == 0 || map.keyframes.at(referenceId).bad)
        referenceId = map.lastKeyframeId;

    if (shouldInsertKeyframe(frame, tracking, map)) {
        const int kfId = insertKeyframe(map, frame, tracking.T_cw);
        result.keyframeInserted = true;
        result.keyframeId = kfId;
        referenceId = kfId;

        // Sec. VI-A: keyframe insertion (observations, new RGB-D points, covisibility, spanning tree).
        result.associated = associateObservations(map, kfId, tracking);
        result.createdFromDepth =
            static_cast<int>(createLandmarks(map, kfId, unmatchedFeatures(map, kfId)).size());
        updateCovisibility(map, kfId);
        // Sec. VI-B .. VI-E.
        result.culledLandmarks = cullLandmarks(map, kfId);
        result.triangulated = triangulateNewLandmarks(map, kfId);
        result.fused = fuseLandmarksInNeighbors(map, kfId);
        result.localBA = localBundleAdjustment(map, kfId, mode);
        result.culledKeyframes = cullKeyframes(map, kfId);
        // Sec. VII.
        result.loop = runLoopClosing(map, kfId, mode);
    }

    const Keyframe &ref = map.keyframes.at(referenceId);
    const Pose T_cr = result.keyframeInserted ? Pose{} : composePose(tracking.T_cw, inversePose(ref.T_cw));
    map.frameRecords.push_back({frame.timestamp, referenceId, T_cr});
    return result;
}

bool exportMap(const Map &map, const std::string &path) {
    std::ofstream out(path);
    if (!out) return false;
    std::size_t count = 0;
    for (const auto &[id, lm] : map.landmarks)
        if (!lm.bad) ++count;
    out << "ply\nformat ascii 1.0\nelement vertex " << count
        << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    for (const auto &[id, lm] : map.landmarks) {
        if (lm.bad) continue;
        out << lm.position[0] << ' ' << lm.position[1] << ' ' << lm.position[2] << '\n';
    }
    return static_cast<bool>(out);
}

bool exportTrajectory(const Map &map, const std::string &path) {
    std::ofstream out(path);
    if (!out) return false;
    for (const FrameRecord &record : map.frameRecords) {
        // Walk up the spanning tree past culled keyframes (ORB-SLAM System::SaveTrajectoryTUM).
        Pose T_cr = record.T_cr;
        int kfId = record.referenceKeyframeId;
        while (map.keyframes.at(kfId).bad) {
            T_cr = composePose(T_cr, map.keyframes.at(kfId).T_cp);
            kfId = map.keyframes.at(kfId).parentId;
        }
        writeTumLine(out, record.timestamp, composePose(T_cr, map.keyframes.at(kfId).T_cw));
    }
    return static_cast<bool>(out);
}

bool exportKeyframeTrajectory(const Map &map, const std::string &path) {
    std::ofstream out(path);
    if (!out) return false;
    for (const auto &[id, kf] : map.keyframes) {
        if (!kf.bad) writeTumLine(out, kf.frame.timestamp, kf.T_cw);
    }
    return static_cast<bool>(out);
}

} // namespace slam
