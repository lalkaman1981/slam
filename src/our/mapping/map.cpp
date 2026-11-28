// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include "map.hpp"
#include "execution.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "teammate_interfaces.hpp"

namespace slam {

namespace {

constexpr int kNone = -1;
constexpr double kEpipolarChi2 = 3.84; // 95% chi-square, 1 DoF
constexpr int kLocalMapMaxKeyframes = 80;

std::size_t toIndex(int i) { return static_cast<std::size_t>(i); }

const Keypoint &keypointOf(const Keyframe &kf, int idx) { return kf.frame.keypoints[toIndex(idx)]; }

void buildGrid(const Map &map, Keyframe &kf) {
    const MapConfig &cfg = map.config;
    kf.grid.assign(toIndex(cfg.gridCols * cfg.gridRows), {});
    for (std::size_t i = 0; i < kf.frame.keypoints.size(); ++i) {
        const Keypoint &kp = kf.frame.keypoints[i];
        const int cx = static_cast<int>(std::floor(kp.u * cfg.gridCols / map.camera.width));
        const int cy = static_cast<int>(std::floor(kp.v * cfg.gridRows / map.camera.height));
        if (cx < 0 || cy < 0 || cx >= cfg.gridCols || cy >= cfg.gridRows) continue;
        kf.grid[toIndex(cy * cfg.gridCols + cx)].push_back(static_cast<int>(i));
    }
}

int newLandmark(Map &map, const Vec3 &position, int keyframeId, int keypointIdx) {
    const int id = map.nextLandmarkId++;
    Landmark lm;
    lm.id = id;
    lm.position = position;
    lm.firstKeyframeId = keyframeId;
    lm.referenceKeyframeId = keyframeId;
    lm.descriptor = map.keyframes.at(keyframeId).frame.descriptors[toIndex(keypointIdx)];
    map.landmarks.emplace(id, lm);
    addObservation(map, id, keyframeId, keypointIdx);
    updateNormalAndDepth(map, id);
    map.recentLandmarkIds.push_back(id);
    return id;
}

double sceneMedianDepth(const Map &map, const Keyframe &kf) {
    std::vector<double> depths;
    for (int slot : kf.landmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id < 0) continue;
        depths.push_back(transformPoint(kf.T_cw, map.landmarks.at(id).position)[2]);
    }
    if (depths.empty()) return -1.0;
    std::sort(depths.begin(), depths.end());
    return depths[(depths.size() - 1) / 2];
}

Mat3 computeF12(const Keyframe &kf1, const Keyframe &kf2, const CameraIntrinsics &cam) {
    const Mat3 R12 = kf1.T_cw.R * kf2.T_cw.R.transpose();
    const Vec3 t12 = -(R12 * kf2.T_cw.t) + kf1.T_cw.t;
    Mat3 Kinv{};
    Kinv(0, 0) = 1.0 / cam.fx;
    Kinv(0, 2) = -cam.cx / cam.fx;
    Kinv(1, 1) = 1.0 / cam.fy;
    Kinv(1, 2) = -cam.cy / cam.fy;
    Kinv(2, 2) = 1.0;
    return Kinv.transpose() * skew(t12) * R12 * Kinv;
}

bool checkDistEpipolarLine(const Keypoint &kp1, const Keypoint &kp2, const Mat3 &F12, double sigma2) {
    // Epipolar line in the second image: l = x1^T F12.
    const double a = kp1.u * F12(0, 0) + kp1.v * F12(1, 0) + F12(2, 0);
    const double b = kp1.u * F12(0, 1) + kp1.v * F12(1, 1) + F12(2, 1);
    const double c = kp1.u * F12(0, 2) + kp1.v * F12(1, 2) + F12(2, 2);
    const double num = a * kp2.u + b * kp2.v + c;
    const double den = a * a + b * b;
    if (den == 0.0) return false;
    return num * num / den < kEpipolarChi2 * sigma2;
}

// Unassociated keypoint pairs satisfying the epipolar constraint (ORB-SLAM SearchForTriangulation,
// brute force instead of the BoW vocabulary tree). LocalMapping disables the orientation histogram.
std::vector<std::pair<int, int>> searchForTriangulation(const Map &map, const Keyframe &kf1,
                                                        const Keyframe &kf2, const Mat3 &F12) {
    const CameraIntrinsics &cam = map.camera;
    const Vec3 c2 = transformPoint(kf2.T_cw, cameraCenter(kf1.T_cw));
    const double ex = cam.fx * c2[0] / c2[2] + cam.cx;
    const double ey = cam.fy * c2[1] / c2[2] + cam.cy;

    std::vector<bool> used2(kf2.frame.keypoints.size(), false);
    for (std::size_t i = 0; i < used2.size(); ++i)
        used2[i] = resolveLandmarkId(map, kf2.landmarkIds[i]) >= 0;

    std::vector<std::pair<int, int>> pairs;
    for (std::size_t i1 = 0; i1 < kf1.frame.keypoints.size(); ++i1) {
        if (resolveLandmarkId(map, kf1.landmarkIds[i1]) >= 0) continue;
        const Keypoint &kp1 = kf1.frame.keypoints[i1];
        const bool stereo1 = kf1.frame.depths[i1] > 0.0f;
        int bestDist = map.config.descriptorThresholdLow;
        int bestIdx = kNone;
        for (std::size_t i2 = 0; i2 < kf2.frame.keypoints.size(); ++i2) {
            if (used2[i2]) continue;
            const int dist = hammingDistance(kf1.frame.descriptors[i1], kf2.frame.descriptors[i2]);
            if (dist > map.config.descriptorThresholdLow || dist > bestDist) continue;
            const Keypoint &kp2 = kf2.frame.keypoints[i2];
            const bool stereo2 = kf2.frame.depths[i2] > 0.0f;
            if (!stereo1 && !stereo2) {
                const double dx = ex - kp2.u, dy = ey - kp2.v;
                if (dx * dx + dy * dy < 100.0 * map.scaleFactors[toIndex(kp2.octave)]) continue;
            }
            if (checkDistEpipolarLine(kp1, kp2, F12, map.levelSigma2[toIndex(kp2.octave)])) {
                bestIdx = static_cast<int>(i2);
                bestDist = dist;
            }
        }
        if (bestIdx < 0) continue;
        used2[toIndex(bestIdx)] = true;
        pairs.emplace_back(static_cast<int>(i1), bestIdx);
    }

    return pairs;
}

// Linear triangulation from normalized image coordinates (smallest singular vector of A).
bool triangulateDLT(const Vec3 &xn1, const Vec3 &xn2, const Pose &T1, const Pose &T2, Vec3 &X) {
    DenseMatrix A(4, 4);
    auto fillRows = [&A](std::size_t row, const Vec3 &xn, const Pose &T) {
        for (std::size_t c = 0; c < 4; ++c) {
            const double p0 = c < 3 ? T.R(0, c) : T.t[0];
            const double p1 = c < 3 ? T.R(1, c) : T.t[1];
            const double p2 = c < 3 ? T.R(2, c) : T.t[2];
            A(row, c) = xn[0] * p2 - p0;
            A(row + 1, c) = xn[1] * p2 - p1;
        }
    };
    fillRows(0, xn1, T1);
    fillRows(2, xn2, T2);
    DenseMatrix AtA(4, 4);
    for (std::size_t r = 0; r < 4; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            for (std::size_t k = 0; k < 4; ++k)
                AtA(r, c) += A(k, r) * A(k, c);
    std::vector<double> values;
    DenseMatrix vectors;
    symmetricEigen(AtA, values, vectors);
    const double w = vectors(3, 0);
    if (std::abs(w) < 1e-12) return false;
    X = makeVec3(vectors(0, 0) / w, vectors(1, 0) / w, vectors(2, 0) / w);
    return true;
}

} // namespace

Map createMap(const CameraIntrinsics &camera, const MapConfig &config) {
    Map map;
    map.camera = camera;
    map.config = config;
    map.scaleFactors.resize(toIndex(config.levels));
    map.levelSigma2.resize(toIndex(config.levels));
    map.invLevelSigma2.resize(toIndex(config.levels));
    double s = 1.0;
    for (std::size_t l = 0; l < map.scaleFactors.size(); ++l) {
        map.scaleFactors[l] = s;
        map.levelSigma2[l] = s * s;
        map.invLevelSigma2[l] = 1.0 / (s * s);
        s *= config.scaleFactor;
    }
    return map;
}

// ---------------------------------------------------------------------------------------------
// Map maintenance helpers
// ---------------------------------------------------------------------------------------------

int resolveLandmarkId(const Map &map, int landmarkId) {
    for (std::size_t guard = 0; landmarkId >= 0 && guard < map.landmarks.size(); ++guard) {
        const auto it = map.landmarks.find(landmarkId);
        if (it == map.landmarks.end()) return kNone;
        if (!it->second.bad) return landmarkId;
        landmarkId = it->second.replacedBy;
    }
    return kNone;
}

void addObservation(Map &map, int landmarkId, int keyframeId, int keypointIdx) {
    map.landmarks.at(landmarkId).observations[keyframeId] = keypointIdx;
    map.keyframes.at(keyframeId).landmarkIds[toIndex(keypointIdx)] = landmarkId;
}

void eraseObservation(Map &map, int landmarkId, int keyframeId) {
    Landmark &lm = map.landmarks.at(landmarkId);
    const auto it = lm.observations.find(keyframeId);
    if (it == lm.observations.end()) return;
    Keyframe &kf = map.keyframes.at(keyframeId);
    if (kf.landmarkIds[toIndex(it->second)] == landmarkId) kf.landmarkIds[toIndex(it->second)] = kNone;
    lm.observations.erase(it);
    if (lm.referenceKeyframeId == keyframeId && !lm.observations.empty())
        lm.referenceKeyframeId = lm.observations.begin()->first;
    // Sec. VI-B: a point observed from fewer than three keyframes is removed.
    if (static_cast<int>(lm.observations.size()) < map.config.minObservingKeyframes)
        setLandmarkBad(map, landmarkId);
}

void setLandmarkBad(Map &map, int landmarkId) {
    Landmark &lm = map.landmarks.at(landmarkId);
    lm.bad = true;
    for (const auto &[kfId, idx] : lm.observations) {
        Keyframe &kf = map.keyframes.at(kfId);
        if (kf.landmarkIds[toIndex(idx)] == landmarkId) kf.landmarkIds[toIndex(idx)] = kNone;
    }
    lm.observations.clear();
}

void replaceLandmark(Map &map, int oldId, int newId) {
    if (oldId == newId) return;
    Landmark &oldLm = map.landmarks.at(oldId);
    Landmark &newLm = map.landmarks.at(newId);
    const std::map<int, int> observations = oldLm.observations;
    oldLm.observations.clear();
    oldLm.bad = true;
    oldLm.replacedBy = newId;
    for (const auto &[kfId, idx] : observations) {
        if (newLm.observations.count(kfId) == 0) {
            addObservation(map, newId, kfId, idx);
        } else {
            Keyframe &kf = map.keyframes.at(kfId);
            if (kf.landmarkIds[toIndex(idx)] == oldId) kf.landmarkIds[toIndex(idx)] = kNone;
        }
    }
    newLm.foundCount += oldLm.foundCount;
    newLm.visibleCount += oldLm.visibleCount;
    chooseRepresentativeDescriptor(map, newId);
}

void setKeyframeBad(Map &map, int keyframeId) {
    Keyframe &kf = map.keyframes.at(keyframeId);
    if (kf.bad || keyframeId == map.originKeyframeId || !kf.loopEdgeIds.empty() || kf.parentId < 0) return;

    for (const auto &[otherId, w] : kf.covisibilityWeights)
        map.keyframes.at(otherId).covisibilityWeights.erase(keyframeId);
    for (std::size_t i = 0; i < kf.landmarkIds.size(); ++i) {
        const int id = kf.landmarkIds[i];
        if (id >= 0 && !map.landmarks.at(id).bad) eraseObservation(map, id, keyframeId);
    }
    kf.covisibilityWeights.clear();

    // Reassign children in the spanning tree: each child takes the most covisible keyframe among
    // the parent and already reassigned children (ORB-SLAM KeyFrame::SetBadFlag).
    std::set<int> parentCandidates{kf.parentId};
    std::set<int> children = kf.childIds;
    while (!children.empty()) {
        int bestWeight = -1, bestChild = kNone, bestParent = kNone;
        for (int childId : children) {
            const Keyframe &child = map.keyframes.at(childId);
            if (child.bad) continue;
            for (int candidate : covisibleKeyframes(map, childId)) {
                const int w = child.covisibilityWeights.at(candidate);
                if (parentCandidates.count(candidate) != 0 && w > bestWeight) {
                    bestWeight = w;
                    bestChild = childId;
                    bestParent = candidate;
                }
            }
        }
        if (bestChild == kNone) break;
        map.keyframes.at(bestChild).parentId = bestParent;
        map.keyframes.at(bestParent).childIds.insert(bestChild);
        parentCandidates.insert(bestChild);
        children.erase(bestChild);
    }
    for (int childId : children) {
        map.keyframes.at(childId).parentId = kf.parentId;
        map.keyframes.at(kf.parentId).childIds.insert(childId);
    }
    Keyframe &parent = map.keyframes.at(kf.parentId);
    parent.childIds.erase(keyframeId);
    kf.childIds.clear();
    kf.T_cp = composePose(kf.T_cw, inversePose(parent.T_cw));
    kf.bad = true;
}

void updateNormalAndDepth(Map &map, int landmarkId) {
    Landmark &lm = map.landmarks.at(landmarkId);
    if (lm.bad || lm.observations.empty()) return;
    if (lm.observations.count(lm.referenceKeyframeId) == 0)
        lm.referenceKeyframeId = lm.observations.begin()->first;

    Vec3 normal{};
    std::size_t directions = 0;
    for (const auto &[kfId, idx] : lm.observations) {
        const Vec3 dir = lm.position - cameraCenter(map.keyframes.at(kfId).T_cw);
        const double n = norm(dir);
        if (n > 0.0) {
            normal += (1.0 / n) * dir;
            ++directions;
        }
    }
    // Average unit rays, preserving their angular spread for the viewing-angle test.
    lm.normal = directions ? (1.0 / static_cast<double>(directions)) * normal : Vec3{};

    const Keyframe &ref = map.keyframes.at(lm.referenceKeyframeId);
    const double dist = norm(lm.position - cameraCenter(ref.T_cw));
    const int level = keypointOf(ref, lm.observations.at(lm.referenceKeyframeId)).octave;
    lm.maxDistance = dist * map.scaleFactors[toIndex(level)];
    lm.minDistance = lm.maxDistance / map.scaleFactors.back();
}

int predictScaleLevel(const Map &map, const Landmark &landmark, double distance) {
    const double ratio = landmark.maxDistance / distance;
    const int level = static_cast<int>(std::ceil(std::log(ratio) / std::log(map.config.scaleFactor)));
    return std::clamp(level, 0, map.config.levels - 1);
}

std::vector<int> connectedKeyframes(const Map &map, int keyframeId) {
    std::vector<std::pair<int, int>> weighted;
    for (const auto &[otherId, w] : map.keyframes.at(keyframeId).covisibilityWeights) {
        if (w <= 0 || map.keyframes.at(otherId).bad) continue;
        weighted.emplace_back(w, otherId);
    }
    std::sort(weighted.begin(), weighted.end(), [](const auto &a, const auto &b) {
        return a.first != b.first ? a.first > b.first : a.second < b.second;
    });
    std::vector<int> ids;
    ids.reserve(weighted.size());
    for (const auto &[w, id] : weighted)
        ids.push_back(id);
    return ids;
}

std::vector<int> covisibleKeyframes(const Map &map, int keyframeId, int minWeight) {
    auto ids = connectedKeyframes(map, keyframeId);
    const auto &weights = map.keyframes.at(keyframeId).covisibilityWeights;
    const int threshold = minWeight > 0 ? minWeight : map.config.covisibilityMinShared;
    const auto end = std::find_if(ids.begin(), ids.end(),
                                  [&](int id) { return weights.at(id) < threshold; });
    // Default searches retain one strongest neighbor if none reaches the threshold.
    // Explicit thresholds (e.g. the Essential Graph) have no fallback.
    if (minWeight == 0 && end == ids.begin() && !ids.empty())
        ids.resize(1);
    else
        ids.erase(end, ids.end());
    return ids;
}

std::vector<int> bestCovisibleKeyframes(const Map &map, int keyframeId, std::size_t n) {
    std::vector<int> ids = covisibleKeyframes(map, keyframeId);
    if (ids.size() > n) ids.resize(n);
    return ids;
}

int trackedLandmarks(const Map &map, int keyframeId, int minObservations) {
    int count = 0;
    for (int slot : map.keyframes.at(keyframeId).landmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id < 0) continue;
        if (static_cast<int>(map.landmarks.at(id).observations.size()) >= minObservations) ++count;
    }
    return count;
}

std::vector<int> featuresInArea(const Map &map, const Keyframe &keyframe, double u, double v, double radius,
                                int minLevel, int maxLevel) {
    std::vector<int> indices;
    const MapConfig &cfg = map.config;
    const double cellW = static_cast<double>(map.camera.width) / cfg.gridCols;
    const double cellH = static_cast<double>(map.camera.height) / cfg.gridRows;
    const int minX = std::max(0, static_cast<int>(std::floor((u - radius) / cellW)));
    const int maxX = std::min(cfg.gridCols - 1, static_cast<int>(std::floor((u + radius) / cellW)));
    const int minY = std::max(0, static_cast<int>(std::floor((v - radius) / cellH)));
    const int maxY = std::min(cfg.gridRows - 1, static_cast<int>(std::floor((v + radius) / cellH)));
    for (int cy = minY; cy <= maxY; ++cy) {
        for (int cx = minX; cx <= maxX; ++cx) {
            for (int idx : keyframe.grid[toIndex(cy * cfg.gridCols + cx)]) {
                const Keypoint &kp = keypointOf(keyframe, idx);
                if (minLevel > 0 && kp.octave < minLevel) continue;
                if (maxLevel >= 0 && kp.octave > maxLevel) continue;
                if (std::abs(kp.u - u) < radius && std::abs(kp.v - v) < radius) indices.push_back(idx);
            }
        }
    }
    return indices;
}

std::vector<int> unmatchedFeatures(const Map &map, int keyframeId) {
    std::vector<int> unmatched;
    const Keyframe &kf = map.keyframes.at(keyframeId);
    for (std::size_t i = 0; i < kf.landmarkIds.size(); ++i)
        if (resolveLandmarkId(map, kf.landmarkIds[i]) < 0) unmatched.push_back(static_cast<int>(i));
    return unmatched;
}

// ---------------------------------------------------------------------------------------------
// Plan functions
// ---------------------------------------------------------------------------------------------

int initializeMap(Map &map, const Frame &firstFrame) {
    if (!map.keyframes.empty()) throw std::logic_error("initializeMap: map is already initialized");
    const auto validDepth =
        std::count_if(firstFrame.depths.begin(), firstFrame.depths.end(), [](float d) { return d > 0.0f; });
    if (validDepth < map.config.minInitialPoints) return kNone;

    // The first frame defines the world origin (ORB-SLAM2 StereoInitialization).
    const int kfId = insertKeyframe(map, firstFrame, Pose{});
    for (std::size_t i = 0; i < firstFrame.keypoints.size(); ++i) {
        const float d = firstFrame.depths[i];
        if (d <= 0.0f) continue;
        const Keypoint &kp = firstFrame.keypoints[i];
        newLandmark(map, backProject(kp.u, kp.v, d, map.camera), kfId, static_cast<int>(i));
    }
    map.frameRecords.push_back({firstFrame.timestamp, kfId, Pose{}});
    return kfId;
}

bool shouldInsertKeyframe(const Frame &frame, const TrackingResult &tracking, const Map &map) {
    const MapConfig &cfg = map.config;
    if (!tracking.ok || map.keyframes.empty() || !cfg.localMappingIdle) return false;
    // Culled keyframes remain as trajectory references, not active map members.
    const int nKeyframes = static_cast<int>(std::count_if(
        map.keyframes.begin(), map.keyframes.end(), [](const auto &entry) { return !entry.second.bad; }));

    // Condition 1: more than 20 frames since the last relocalization.
    if (map.lastRelocalizationFrameId >= 0 &&
        frame.id < map.lastRelocalizationFrameId + cfg.maxFramesAfterRelocalization &&
        nKeyframes > cfg.maxFramesAfterRelocalization)
        return false;

    int tracked = 0, trackedClose = 0, untrackedClose = 0;
    for (std::size_t i = 0; i < frame.keypoints.size(); ++i) {
        const bool hasLandmark =
            i < tracking.landmarkIds.size() && resolveLandmarkId(map, tracking.landmarkIds[i]) >= 0;
        if (hasLandmark) ++tracked;
        const float d = i < frame.depths.size() ? frame.depths[i] : 0.0f;
        if (d > 0.0f && d < cfg.closeDepth) {
            if (hasLandmark)
                ++trackedClose;
            else
                ++untrackedClose;
        }
    }

    int refId = tracking.referenceKeyframeId;
    if (map.keyframes.count(refId) == 0 || map.keyframes.at(refId).bad) refId = map.lastKeyframeId;
    // Single initial RGB-D keyframe: its points have one observation each.
    const int minObs = nKeyframes == 1 ? 1 : (nKeyframes == 2 ? 2 : 3);
    const int refTracked = trackedLandmarks(map, refId, minObs);

    const bool needClose = cfg.useCloseDepthRule && trackedClose < cfg.closeTrackedLimit &&
                           untrackedClose > cfg.closeUntrackedLimit;
    // Condition 2: local mapping idle, or more than 20 frames since the last keyframe.
    const bool c1a = frame.id >= map.lastKeyframeFrameId + cfg.maxFramesBetweenKeyframes;
    const bool c1b = cfg.localMappingIdle;
    // Conditions 3 and 4: at least 50 tracked points, and fewer than 90% of K_ref's points.
    const bool c2 =
        (tracked < cfg.referenceTrackedRatio * refTracked || needClose) && tracked > cfg.minTrackedPoints;
    return (c1a || c1b) && c2;
}

int insertKeyframe(Map &map, const Frame &frame, const Pose &T_cw) {
    const int id = map.nextKeyframeId++;
    Keyframe kf;
    kf.id = id;
    kf.frame = frame;
    if (kf.frame.depths.size() != kf.frame.keypoints.size())
        kf.frame.depths.resize(kf.frame.keypoints.size(), 0.0f);
    kf.T_cw = T_cw;
    kf.landmarkIds.assign(frame.keypoints.size(), kNone);
    buildGrid(map, kf);
    map.keyframes.emplace(id, std::move(kf));
    if (map.originKeyframeId < 0) map.originKeyframeId = id;
    map.lastKeyframeId = id;
    map.lastKeyframeFrameId = frame.id;
    return id;
}

LocalMap selectLocalMap(const Map &map, const std::vector<int> &trackedLandmarkIds) {
    // Sec. V-D: K1 shares points with the frame, K2 are neighbours of K1 in the covisibility graph.
    LocalMap local;
    std::map<int, int> counter;
    for (int slot : trackedLandmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id < 0) continue;
        for (const auto &[kfId, idx] : map.landmarks.at(id).observations)
            ++counter[kfId];
    }
    if (counter.empty()) return local;

    std::set<int> included;
    int best = -1;
    for (const auto &[kfId, count] : counter) {
        if (map.keyframes.at(kfId).bad) continue;
        local.keyframeIds.push_back(kfId);
        included.insert(kfId);
        if (count > best) {
            best = count;
            local.referenceKeyframeId = kfId;
        }
    }

    const std::vector<int> k1 = local.keyframeIds;
    for (int kfId : k1) {
        if (static_cast<int>(local.keyframeIds.size()) > kLocalMapMaxKeyframes) break;
        auto tryAdd = [&](int id) {
            if (id < 0 || included.count(id) != 0 || map.keyframes.at(id).bad) return false;
            local.keyframeIds.push_back(id);
            included.insert(id);
            return true;
        };
        for (int n : bestCovisibleKeyframes(map, kfId, 10))
            if (tryAdd(n)) break;
        for (int c : map.keyframes.at(kfId).childIds)
            if (tryAdd(c)) break;
        tryAdd(map.keyframes.at(kfId).parentId);
    }

    std::set<int> seen;
    for (int kfId : local.keyframeIds) {
        for (int slot : map.keyframes.at(kfId).landmarkIds) {
            const int id = resolveLandmarkId(map, slot);
            if (id >= 0 && seen.insert(id).second) local.landmarkIds.push_back(id);
        }
    }
    return local;
}

std::vector<VisibleLandmark> findVisibleLandmarks(const Map &map, const LocalMap &localMap,
                                                  const Pose &predictedPose, const CameraIntrinsics &camera,
                                                  ExecutionMode mode) {
    std::vector<int> ids;
    std::vector<Vec3> positions;
    for (int slot : localMap.landmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id < 0) continue;
        ids.push_back(id);
        positions.push_back(map.landmarks.at(id).position);
    }
    std::vector<Vec3> cameraPoints(ids.size());
    detail::runPartitions(ids.size(), detail::workerCount(ids.size(), mode),
                          [&](std::size_t begin, std::size_t end, std::size_t) {
                              std::size_t i = begin;
                              if (mode == ExecutionMode::SerialSimd) {
                                  for (; i + 1 < end; i += 2) {
                                      const auto pair = detail::transformPair(predictedPose, predictedPose,
                                                                              positions[i], positions[i + 1]);
                                      for (std::size_t q = 0; q < 3; ++q) {
                                          double lanes[2];
                                          pair[q].store(lanes);
                                          cameraPoints[i][q] = lanes[0];
                                          cameraPoints[i + 1][q] = lanes[1];
                                      }
                                  }
                              }
                              for (; i < end; ++i)
                                  cameraPoints[i] = transformPoint(predictedPose, positions[i]);
                          });
    const Vec3 Ow = cameraCenter(predictedPose);
    std::vector<VisibleLandmark> candidates(ids.size());
    std::vector<unsigned char> keep(ids.size(), 0);
    detail::runPartitions(ids.size(), detail::workerCount(ids.size(), mode),
                          [&](std::size_t begin, std::size_t end, std::size_t) {
                              for (std::size_t i = begin; i < end; ++i) {
                                  const int id = ids[i];
                                  const Landmark &lm = map.landmarks.at(id);
                                  double u = 0.0, v = 0.0;
                                  if (!project(cameraPoints[i], camera, u, v)) continue;
                                  const Vec3 PO = lm.position - Ow;
                                  const double dist = norm(PO);
                                  if (dist <= 0.0 || dist < 0.8 * lm.minDistance ||
                                      dist > 1.2 * lm.maxDistance)
                                      continue;
                                  const double viewCos = dot(PO, lm.normal) / dist;
                                  if (viewCos < 0.5) continue;
                                  candidates[i] = {id, u, v, predictScaleLevel(map, lm, dist), viewCos};
                                  keep[i] = 1;
                              }
                          });
    std::vector<VisibleLandmark> visible;
    for (std::size_t i = 0; i < ids.size(); ++i)
        if (keep[i]) visible.push_back(candidates[i]);

    return visible;
}

int associateObservations(Map &map, int keyframeId, const TrackingResult &tracking) {
    Keyframe &kf = map.keyframes.at(keyframeId);
    const std::size_t n = std::min(tracking.landmarkIds.size(), kf.landmarkIds.size());
    int associated = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const int id = resolveLandmarkId(map, tracking.landmarkIds[i]);
        if (id < 0 || map.landmarks.at(id).observations.count(keyframeId) != 0) continue;
        addObservation(map, id, keyframeId, static_cast<int>(i));
        updateNormalAndDepth(map, id);
        chooseRepresentativeDescriptor(map, id);
        ++associated;
    }
    return associated;
}

std::vector<int> createLandmarks(Map &map, int keyframeId, const std::vector<int> &unmatched) {
    // ORB-SLAM2 RGB-D: create points from the closest depths first; keep going past the close
    // threshold until at least minNewPointsPerKeyframe points are associated.
    const Keyframe &kf = map.keyframes.at(keyframeId);
    const std::set<int> candidates(unmatched.begin(), unmatched.end());
    std::vector<int> order;
    for (std::size_t i = 0; i < kf.frame.depths.size(); ++i)
        if (kf.frame.depths[i] > 0.0f) order.push_back(static_cast<int>(i));
    std::sort(order.begin(), order.end(),
              [&kf](int a, int b) { return kf.frame.depths[toIndex(a)] < kf.frame.depths[toIndex(b)]; });

    const Pose T_wc = inversePose(kf.T_cw);
    std::vector<int> created;
    int nPoints = 0;
    for (int idx : order) {
        const float depth = map.keyframes.at(keyframeId).frame.depths[toIndex(idx)];
        if (resolveLandmarkId(map, map.keyframes.at(keyframeId).landmarkIds[toIndex(idx)]) >= 0) {
            ++nPoints;
        } else if (candidates.count(idx) != 0) {
            const Keypoint &kp = keypointOf(map.keyframes.at(keyframeId), idx);
            const Vec3 Xw = transformPoint(T_wc, backProject(kp.u, kp.v, depth, map.camera));
            created.push_back(newLandmark(map, Xw, keyframeId, idx));
            ++nPoints;
        }
        if (depth > map.config.closeDepth && nPoints > map.config.minNewPointsPerKeyframe) break;
    }
    return created;
}

void chooseRepresentativeDescriptor(Map &map, int landmarkId) {
    // Sec. III-C: the descriptor whose median Hamming distance to all others is minimal.
    Landmark &lm = map.landmarks.at(landmarkId);
    if (lm.bad || lm.observations.empty()) return;
    std::vector<const Descriptor *> descriptors;
    for (const auto &[kfId, idx] : lm.observations) {
        const Keyframe &kf = map.keyframes.at(kfId);
        if (!kf.bad) descriptors.push_back(&kf.frame.descriptors[toIndex(idx)]);
    }
    if (descriptors.empty()) return;
    const std::size_t n = descriptors.size();
    std::vector<int> distances(n * n, 0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            distances[i * n + j] = distances[j * n + i] = hammingDistance(*descriptors[i], *descriptors[j]);
    int bestMedian = std::numeric_limits<int>::max();
    std::size_t bestIdx = 0;
    std::vector<int> row(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::copy(distances.begin() + static_cast<std::ptrdiff_t>(i * n),
                  distances.begin() + static_cast<std::ptrdiff_t>((i + 1) * n), row.begin());
        std::sort(row.begin(), row.end());
        const int median = row[(n - 1) / 2];
        if (median < bestMedian) {
            bestMedian = median;
            bestIdx = i;
        }
    }
    lm.descriptor = *descriptors[bestIdx];
}

int cullLandmarks(Map &map, int currentKeyframeId) {
    // Sec. VI-B: during its first three keyframes a new point must be found in more than 25% of
    // the frames where it is predicted visible and, after one keyframe, be seen by three keyframes.
    const MapConfig &cfg = map.config;
    int culled = 0;
    std::vector<int> keep;
    for (int id : map.recentLandmarkIds) {
        Landmark &lm = map.landmarks.at(id);
        const int age = currentKeyframeId - lm.firstKeyframeId;
        if (lm.bad) continue;
        if (static_cast<double>(lm.foundCount) / lm.visibleCount < cfg.minFoundRatio) {
            setLandmarkBad(map, id);
            ++culled;
        } else if (age >= 2 && static_cast<int>(lm.observations.size()) < cfg.minObservingKeyframes) {
            setLandmarkBad(map, id);
            ++culled;
        } else if (age < 3) {
            keep.push_back(id);
        }
    }
    map.recentLandmarkIds = std::move(keep);
    return culled;
}

void updateCovisibility(Map &map, int keyframeId) {
    Keyframe &kf = map.keyframes.at(keyframeId);
    std::map<int, int> counter;
    for (int slot : kf.landmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id < 0) continue;
        for (const auto &[otherId, idx] : map.landmarks.at(id).observations) {
            if (otherId != keyframeId && !map.keyframes.at(otherId).bad) ++counter[otherId];
        }
    }

    for (const auto &[otherId, w] : kf.covisibilityWeights)
        if (counter.count(otherId) == 0) map.keyframes.at(otherId).covisibilityWeights.erase(keyframeId);
    kf.covisibilityWeights = counter;
    if (counter.empty()) return;

    int maxCount = 0, maxId = kNone;
    for (const auto &[otherId, count] : counter) {
        if (count > maxCount) {
            maxCount = count;
            maxId = otherId;
        }
        // Retain every shared count; strong-neighbor selection is separate.
        map.keyframes.at(otherId).covisibilityWeights[keyframeId] = count;
    }
    // Spanning tree: a new keyframe hangs from the keyframe sharing most points (Sec. III-E).
    if (kf.parentId < 0 && keyframeId != map.originKeyframeId) {
        kf.parentId = maxId;
        map.keyframes.at(maxId).childIds.insert(keyframeId);
    }
}

int triangulateNewLandmarks(Map &map, int keyframeId) {
    // Sec. VI-C: triangulate unmatched ORB features with the most covisible keyframes.
    const MapConfig &cfg = map.config;
    const CameraIntrinsics &cam = map.camera;
    const std::vector<int> neighbors =
        bestCovisibleKeyframes(map, keyframeId, toIndex(cfg.triangulationNeighbors));
    const double ratioFactor = 1.5 * cfg.scaleFactor;
    int created = 0;

    for (int otherId : neighbors) {
        const Keyframe &kf1 = map.keyframes.at(keyframeId);
        const Keyframe &kf2 = map.keyframes.at(otherId);
        const Vec3 Ow1 = cameraCenter(kf1.T_cw);
        const Vec3 Ow2 = cameraCenter(kf2.T_cw);
        const double medianDepth = sceneMedianDepth(map, kf2);
        if (medianDepth <= 0.0 || norm(Ow2 - Ow1) / medianDepth < cfg.minBaselineDepthRatio) continue;

        const Mat3 F12 = computeF12(kf1, kf2, cam);
        const auto pairs = searchForTriangulation(map, kf1, kf2, F12);
        const Mat3 Rwc1 = kf1.T_cw.R.transpose();
        const Mat3 Rwc2 = kf2.T_cw.R.transpose();

        for (const auto &[i1, i2] : pairs) {
            if (resolveLandmarkId(map, map.keyframes.at(keyframeId).landmarkIds[toIndex(i1)]) >= 0) continue;
            const Keypoint &kp1 = keypointOf(kf1, i1);
            const Keypoint &kp2 = keypointOf(kf2, i2);
            const Vec3 xn1 = makeVec3((kp1.u - cam.cx) / cam.fx, (kp1.v - cam.cy) / cam.fy, 1.0);
            const Vec3 xn2 = makeVec3((kp2.u - cam.cx) / cam.fx, (kp2.v - cam.cy) / cam.fy, 1.0);
            const Vec3 ray1 = Rwc1 * xn1;
            const Vec3 ray2 = Rwc2 * xn2;
            const double cosParallax = dot(ray1, ray2) / (norm(ray1) * norm(ray2));
            if (!(cosParallax > 0.0 && cosParallax < cfg.maxParallaxCos)) continue;

            Vec3 X;
            if (!triangulateDLT(xn1, xn2, kf1.T_cw, kf2.T_cw, X)) continue;
            const Vec3 x1 = transformPoint(kf1.T_cw, X);
            const Vec3 x2 = transformPoint(kf2.T_cw, X);
            if (x1[2] <= 0.0 || x2[2] <= 0.0) continue;

            auto reprojectionOk = [&](const Vec3 &xc, const Keypoint &kp) {
                const double eu = cam.fx * xc[0] / xc[2] + cam.cx - kp.u;
                const double ev = cam.fy * xc[1] / xc[2] + cam.cy - kp.v;
                return eu * eu + ev * ev <= cfg.chi2Mono * map.levelSigma2[toIndex(kp.octave)];
            };
            if (!reprojectionOk(x1, kp1) || !reprojectionOk(x2, kp2)) continue;

            // Scale consistency between the distance ratio and the pyramid level ratio.
            const double dist1 = norm(X - Ow1);
            const double dist2 = norm(X - Ow2);
            if (dist1 == 0.0 || dist2 == 0.0) continue;
            const double ratioDist = dist2 / dist1;
            const double ratioOctave =
                map.scaleFactors[toIndex(kp1.octave)] / map.scaleFactors[toIndex(kp2.octave)];
            if (ratioDist * ratioFactor < ratioOctave || ratioDist > ratioOctave * ratioFactor) continue;

            const int id = newLandmark(map, X, keyframeId, i1);
            addObservation(map, id, otherId, i2);
            chooseRepresentativeDescriptor(map, id);
            updateNormalAndDepth(map, id);
            ++created;
        }
    }
    return created;
}

int fuseLandmarksIntoKeyframe(Map &map, int keyframeId, const std::vector<int> &landmarkIds,
                              double radiusFactor, bool checkReprojection, bool preferIncoming) {
    const CameraIntrinsics &cam = map.camera;
    int fused = 0;
    for (int slot : landmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id < 0) continue;
        const Keyframe &kf = map.keyframes.at(keyframeId);
        const Landmark &lm = map.landmarks.at(id);
        if (lm.observations.count(keyframeId) != 0) continue;

        const Vec3 pointCamera = transformPoint(kf.T_cw, lm.position);
        double u = 0.0, v = 0.0;
        if (!project(pointCamera, cam, u, v)) continue;
        const Vec3 PO = lm.position - cameraCenter(kf.T_cw);
        const double dist = norm(PO);
        if (dist < 0.8 * lm.minDistance || dist > 1.2 * lm.maxDistance) continue;
        if (dot(PO, lm.normal) < 0.5 * dist) continue;

        const int level = predictScaleLevel(map, lm, dist);
        const double radius = radiusFactor * map.scaleFactors[toIndex(level)];
        int bestDist = 256, bestIdx = kNone;
        for (int idx : featuresInArea(map, kf, u, v, radius, -1, -1)) {
            const Keypoint &kp = keypointOf(kf, idx);
            if (kp.octave < level - 1 || kp.octave > level) continue;
            if (checkReprojection) {
                const double eu = u - kp.u, ev = v - kp.v;
                double error2 = eu * eu + ev * ev;
                double threshold = map.config.chi2Mono;
                const float depth = kf.frame.depths[toIndex(idx)];
                if (map.config.useDepthInBundleAdjustment && depth > 0.0f) {
                    // RGB-D uses the same virtual right-image observation as local BA.
                    const double er = eu - cam.bf / pointCamera[2] + cam.bf / depth;
                    error2 += er * er;
                    threshold = map.config.chi2Stereo;
                }
                if (error2 * map.invLevelSigma2[toIndex(kp.octave)] > threshold) continue;
            }
            const int d = hammingDistance(lm.descriptor, kf.frame.descriptors[toIndex(idx)]);
            if (d < bestDist) {
                bestDist = d;
                bestIdx = idx;
            }
        }
        if (bestIdx < 0 || bestDist > map.config.descriptorThresholdLow) continue;

        const int existing = resolveLandmarkId(map, kf.landmarkIds[toIndex(bestIdx)]);
        if (existing >= 0) {
            if (existing == id) continue;
            const bool keepExisting =
                !preferIncoming && map.landmarks.at(existing).observations.size() > lm.observations.size();
            if (keepExisting)
                replaceLandmark(map, id, existing);
            else
                replaceLandmark(map, existing, id);
        } else {
            addObservation(map, id, keyframeId, bestIdx);
        }
        ++fused;
    }
    return fused;
}

int searchByProjection(const Map &map, const Keyframe &keyframe, const Pose &T_cw,
                       const std::vector<int> &landmarkIds, std::vector<int> &matchedLandmarks,
                       double radiusFactor) {
    const CameraIntrinsics &cam = map.camera;
    const Vec3 Ow = cameraCenter(T_cw);
    std::set<int> alreadyMatched;
    for (int id : matchedLandmarks)
        if (id >= 0) alreadyMatched.insert(id);

    int added = 0;
    for (int slot : landmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id < 0 || alreadyMatched.count(id) != 0) continue;
        const Landmark &lm = map.landmarks.at(id);
        double u = 0.0, v = 0.0;
        if (!project(transformPoint(T_cw, lm.position), cam, u, v)) continue;
        const Vec3 PO = lm.position - Ow;
        const double dist = norm(PO);
        if (dist < 0.8 * lm.minDistance || dist > 1.2 * lm.maxDistance) continue;
        if (dot(PO, lm.normal) < 0.5 * dist) continue;

        const int level = predictScaleLevel(map, lm, dist);
        const double radius = radiusFactor * map.scaleFactors[toIndex(level)];
        int bestDist = 256, bestIdx = kNone;
        for (int idx : featuresInArea(map, keyframe, u, v, radius, -1, -1)) {
            if (matchedLandmarks[toIndex(idx)] >= 0) continue;
            const int kpLevel = keypointOf(keyframe, idx).octave;
            if (kpLevel < level - 1 || kpLevel > level) continue;
            const int d = hammingDistance(lm.descriptor, keyframe.frame.descriptors[toIndex(idx)]);
            if (d < bestDist) {
                bestDist = d;
                bestIdx = idx;
            }
        }
        if (bestIdx >= 0 && bestDist <= map.config.descriptorThresholdLow) {
            matchedLandmarks[toIndex(bestIdx)] = id;
            alreadyMatched.insert(id);
            ++added;
        }
    }
    return added;
}

int fuseLandmarksInNeighbors(Map &map, int keyframeId) {
    // Sec. VI-C (end): project the keyframe's points into its neighbours and the neighbours'
    // points into the keyframe, fusing duplicates (ORB-SLAM LocalMapping::SearchInNeighbors).
    const MapConfig &cfg = map.config;
    std::vector<int> targets;
    std::set<int> targetSet{keyframeId};
    for (int n : bestCovisibleKeyframes(map, keyframeId, toIndex(cfg.fuseNeighbors))) {
        if (!targetSet.insert(n).second) continue;
        targets.push_back(n);
    }
    const std::vector<int> firstLevel = targets;
    for (int n : firstLevel) {
        for (int n2 : bestCovisibleKeyframes(map, n, toIndex(cfg.fuseSecondNeighbors))) {
            if (targetSet.insert(n2).second) targets.push_back(n2);
        }
    }

    int fused = 0;
    std::vector<int> ownPoints;
    for (int slot : map.keyframes.at(keyframeId).landmarkIds)
        if (resolveLandmarkId(map, slot) >= 0) ownPoints.push_back(slot);
    for (int t : targets)
        fused += fuseLandmarksIntoKeyframe(map, t, ownPoints, cfg.fuseRadius, true, false);

    std::vector<int> candidates;
    std::set<int> seen;
    for (int t : targets) {
        for (int slot : map.keyframes.at(t).landmarkIds) {
            const int id = resolveLandmarkId(map, slot);
            if (id >= 0 && seen.insert(id).second) candidates.push_back(id);
        }
    }
    fused += fuseLandmarksIntoKeyframe(map, keyframeId, candidates, cfg.fuseRadius, true, false);

    for (int slot : map.keyframes.at(keyframeId).landmarkIds) {
        const int id = resolveLandmarkId(map, slot);
        if (id < 0) continue;
        chooseRepresentativeDescriptor(map, id);
        updateNormalAndDepth(map, id);
    }
    updateCovisibility(map, keyframeId);
    return fused;
}

int cullKeyframes(Map &map, int currentKeyframeId) {
    // Sec. VI-E: discard local keyframes whose points are 90% seen by at least three other
    // keyframes at the same or finer scale.
    const MapConfig &cfg = map.config;
    int culled = 0;
    for (int kfId : covisibleKeyframes(map, currentKeyframeId)) {
        if (kfId == map.originKeyframeId) continue;
        const Keyframe &kf = map.keyframes.at(kfId);
        if (kf.bad) continue;
        int nPoints = 0, nRedundant = 0;
        for (std::size_t i = 0; i < kf.landmarkIds.size(); ++i) {
            const int id = resolveLandmarkId(map, kf.landmarkIds[i]);
            if (id < 0) continue;
            ++nPoints;
            const Landmark &lm = map.landmarks.at(id);
            if (static_cast<int>(lm.observations.size()) <= cfg.redundantObservers) continue;
            const int level = kf.frame.keypoints[i].octave;
            int observers = 0;
            for (const auto &[otherId, idx] : lm.observations) {
                if (otherId == kfId) continue;
                if (keypointOf(map.keyframes.at(otherId), idx).octave <= level + 1) ++observers;
                if (observers >= cfg.redundantObservers) break;
            }
            if (observers >= cfg.redundantObservers) ++nRedundant;
        }
        if (nPoints > 0 && nRedundant > cfg.redundantRatio * nPoints) {
            setKeyframeBad(map, kfId);
            if (map.keyframes.at(kfId).bad) ++culled;
        }
    }
    return culled;
}

} // namespace slam
