// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include "loop_closing.hpp"
#include "execution.hpp"

#include <algorithm>
#include <utility>

namespace slam {

namespace {

constexpr int kDistributionGrid = 4;
constexpr std::size_t kGroupNeighbors = 10;
constexpr double kGroupRetainRatio = 0.75;

int covisibilityWeight(const Map &map, int a, int b) {
    const auto &weights = map.keyframes.at(a).covisibilityWeights;
    const auto it = weights.find(b);
    return it == weights.end() ? 0 : it->second;
}

} // namespace

std::vector<int> findLoopCandidates(Map &map, int keyframeId, ExecutionMode mode) {
    detail::requireSimdSupport(mode);
    const MapConfig &cfg = map.config;
    if (keyframeId < map.lastLoopKeyframeId + cfg.keyframesBetweenLoops) return {};
    const Keyframe &kf = map.keyframes.at(keyframeId);

    // s_min: lowest similarity to the covisible neighbours with theta_min = 30.
    std::vector<int> neighbours = covisibleKeyframes(map, keyframeId, cfg.loopNeighborMinShared);
    if (neighbours.empty()) neighbours = covisibleKeyframes(map, keyframeId);
    if (neighbours.empty()) {
        map.consistentGroups.clear();
        return {};
    }
    double minScore = 1.0;
    for (int n : neighbours)
        minScore = std::min(minScore, scoreKeyframeSimilarity(kf.frame, map.keyframes.at(n).frame));

    // Brute-force "database" query over older keyframes not connected to the current one.
    std::set<int> connected{keyframeId};
    for (const auto &[otherId, w] : kf.covisibilityWeights)
        connected.insert(otherId);
    std::vector<int> candidateIds;
    for (const auto &[otherId, other] : map.keyframes)
        if (!other.bad && connected.count(otherId) == 0 && otherId <= keyframeId - cfg.loopMinKeyframeGap)
            candidateIds.push_back(otherId);
    std::vector<double> candidateScores(candidateIds.size());
    detail::runPartitions(candidateIds.size(), detail::workerCount(candidateIds.size(), mode, 1),
                          [&](std::size_t begin, std::size_t end, std::size_t) {
                              for (std::size_t i = begin; i < end; ++i)
                                  candidateScores[i] = scoreKeyframeSimilarity(
                                      kf.frame, map.keyframes.at(candidateIds[i]).frame);
                          });
    std::map<int, double> scores;
    for (std::size_t i = 0; i < candidateIds.size(); ++i)
        if (candidateScores[i] >= minScore) scores[candidateIds[i]] = candidateScores[i];
    if (scores.empty()) {
        map.consistentGroups.clear();
        return {};
    }

    // Accumulate scores over covisibility groups and keep groups above 75% of the best.
    std::vector<std::pair<double, int>> groups;
    double bestAccumulated = 0.0;
    for (const auto &[candidateId, score] : scores) {
        double accumulated = score, best = score;
        int bestId = candidateId;
        for (int n : bestCovisibleKeyframes(map, candidateId, kGroupNeighbors)) {
            const auto it = scores.find(n);
            if (it == scores.end()) continue;
            accumulated += it->second;
            if (it->second > best) {
                best = it->second;
                bestId = n;
            }
        }
        groups.emplace_back(accumulated, bestId);
        bestAccumulated = std::max(bestAccumulated, accumulated);
    }
    std::vector<int> representatives;
    std::set<int> added;
    for (const auto &[accumulated, bestId] : groups) {
        if (accumulated > kGroupRetainRatio * bestAccumulated && added.insert(bestId).second)
            representatives.push_back(bestId);
    }

    // Temporal consistency: a candidate group must overlap groups from previous keyframes
    // three consecutive times.
    std::vector<ConsistentGroup> currentGroups;
    std::vector<bool> groupExtended(map.consistentGroups.size(), false);
    std::vector<int> accepted;
    for (int candidateId : representatives) {
        std::set<int> group{candidateId};
        for (int n : covisibleKeyframes(map, candidateId))
            group.insert(n);
        bool enough = false, consistentWithSome = false;
        for (std::size_t g = 0; g < map.consistentGroups.size(); ++g) {
            const std::set<int> &previous = map.consistentGroups[g].keyframeIds;
            const bool overlaps = std::any_of(group.begin(), group.end(),
                                              [&previous](int id) { return previous.count(id) != 0; });
            if (!overlaps) continue;
            consistentWithSome = true;
            const int consistency = map.consistentGroups[g].consistency + 1;
            if (!groupExtended[g]) {
                currentGroups.push_back({group, consistency});
                groupExtended[g] = true;
            }
            if (consistency >= cfg.loopConsistency && !enough) {
                accepted.push_back(candidateId);
                enough = true;
            }
        }
        if (!consistentWithSome) currentGroups.push_back({group, 0});
    }
    map.consistentGroups = std::move(currentGroups);
    return accepted;
}

LoopGeometry computeLoopGeometry(const Map &map, int keyframeId, int candidateId, ExecutionMode mode) {
    LoopGeometry result;
    result.candidateKeyframeId = candidateId;
    const Keyframe &current = map.keyframes.at(keyframeId);
    const Keyframe &candidate = map.keyframes.at(candidateId);

    // Matches between current keypoints and the candidate's map points.
    std::vector<Match> matches;
    for (const Match &m : matchFrames(current.frame, candidate.frame, mode)) {
        if (resolveLandmarkId(map, candidate.landmarkIds[static_cast<std::size_t>(m.trainIdx)]) >= 0)
            matches.push_back(m);
    }
    if (static_cast<int>(matches.size()) < map.config.loopMinMatchesForGeometry) return result;

    result.geometry = verifyLoopGeometry(current.frame, candidate.frame, matches, map.camera, mode);
    if (!result.geometry.success) return result;
    result.correctedT_cw = composePose(result.geometry.T_current_candidate, candidate.T_cw);

    result.matchedLandmarkIds.assign(current.frame.keypoints.size(), -1);
    for (std::size_t m = 0; m < matches.size(); ++m) {
        if (!result.geometry.inliers[m]) continue;
        result.matchedLandmarkIds[static_cast<std::size_t>(matches[m].queryIdx)] =
            resolveLandmarkId(map, candidate.landmarkIds[static_cast<std::size_t>(matches[m].trainIdx)]);
        ++result.inlierMatches;
    }

    // Guided search: project the points of the candidate and its neighbours with the corrected pose.
    std::set<int> seen;
    std::vector<int> loopKeyframes = covisibleKeyframes(map, candidateId);
    loopKeyframes.push_back(candidateId);
    for (int kfId : loopKeyframes) {
        for (int slot : map.keyframes.at(kfId).landmarkIds) {
            const int id = resolveLandmarkId(map, slot);
            if (id >= 0 && seen.insert(id).second) result.loopLandmarkIds.push_back(id);
        }
    }
    searchByProjection(map, current, result.correctedT_cw, result.loopLandmarkIds, result.matchedLandmarkIds,
                       map.config.loopProjectionRadius);
    result.totalMatches = static_cast<int>(std::count_if(
        result.matchedLandmarkIds.begin(), result.matchedLandmarkIds.end(), [](int id) { return id >= 0; }));
    result.success = true;
    return result;
}

bool confirmLoopCandidate(const Map &map, int keyframeId, const LoopGeometry &geometry) {
    const MapConfig &cfg = map.config;
    if (!geometry.success) return false;
    if (geometry.inlierMatches < cfg.loopMinInliers) return false;
    if (geometry.totalMatches < cfg.loopMinTotalMatches) return false;
    if (geometry.geometry.meanError > cfg.loopMaxMeanError) return false;

    const Keyframe &kf = map.keyframes.at(keyframeId);
    std::set<int> cells;
    for (std::size_t i = 0; i < geometry.matchedLandmarkIds.size(); ++i) {
        if (geometry.matchedLandmarkIds[i] < 0) continue;
        const Keypoint &kp = kf.frame.keypoints[i];
        const int cx = std::clamp(static_cast<int>(kp.u * kDistributionGrid / map.camera.width), 0,
                                  kDistributionGrid - 1);
        const int cy = std::clamp(static_cast<int>(kp.v * kDistributionGrid / map.camera.height), 0,
                                  kDistributionGrid - 1);
        cells.insert(cy * kDistributionGrid + cx);
    }
    return static_cast<int>(cells.size()) >= cfg.loopMinOccupiedCells;
}

PoseGraph buildEssentialGraph(const Map &map, int currentId, int loopId,
                              const std::map<int, Pose> &nonCorrected, const std::map<int, Pose> &corrected,
                              const std::map<int, std::set<int>> &loopConnections) {
    PoseGraph graph;
    for (const auto &[id, kf] : map.keyframes) {
        if (kf.bad) continue;
        const auto it = corrected.find(id);
        addGraphNode(graph, id, it != corrected.end() ? it->second : kf.T_cw, id == map.originKeyframeId);
    }
    auto initial = [&graph](int id) { return graph.poses[graph.nodeIndex.at(id)]; };
    auto beforeCorrection = [&](int id) {
        const auto it = nonCorrected.find(id);
        return it != nonCorrected.end() ? it->second : initial(id);
    };
    auto hasNode = [&graph](int id) { return graph.nodeIndex.count(id) != 0; };
    const int minShared = map.config.essentialMinShared;

    // New links created by the loop fusion, measured with the corrected poses.
    std::set<std::pair<int, int>> inserted;
    for (const auto &[i, links] : loopConnections) {
        if (!hasNode(i)) continue;
        const Pose Twi = inversePose(initial(i));
        for (int j : links) {
            if (!hasNode(j)) continue;
            if ((i != currentId || j != loopId) && covisibilityWeight(map, i, j) < minShared) continue;
            addLoopEdge(graph, j, i, composePose(initial(j), Twi));
            inserted.insert(std::minmax(i, j));
        }
    }

    // Spanning tree, previous loop edges and strong covisibility edges, measured before correction.
    for (const auto &[id, kf] : map.keyframes) {
        if (kf.bad) continue;
        const Pose Twi = inversePose(beforeCorrection(id));
        if (kf.parentId >= 0 && hasNode(kf.parentId))
            addOdometryEdge(graph, kf.parentId, id, composePose(beforeCorrection(kf.parentId), Twi));
        for (int l : kf.loopEdgeIds) {
            if (l < id && hasNode(l)) addLoopEdge(graph, l, id, composePose(beforeCorrection(l), Twi));
        }
        for (int n : covisibleKeyframes(map, id, minShared)) {
            if (n == kf.parentId || kf.childIds.count(n) != 0 || kf.loopEdgeIds.count(n) != 0) continue;
            if (n >= id || !hasNode(n) || inserted.count(std::minmax(id, n)) != 0) continue;
            addCovisibilityEdge(graph, n, id, composePose(beforeCorrection(n), Twi));
        }
    }
    return graph;
}

void applyMapCorrection(Map &map, const std::map<int, Pose> &oldPoses, const std::map<int, Pose> &newPoses,
                        int currentKeyframeId) {
    for (auto &[id, lm] : map.landmarks) {
        if (lm.bad) continue;
        const int ref =
            lm.correctedByKeyframeId == currentKeyframeId ? lm.correctedReferenceId : lm.referenceKeyframeId;
        const auto oldIt = oldPoses.find(ref);
        const auto newIt = newPoses.find(ref);
        if (oldIt == oldPoses.end() || newIt == newPoses.end()) continue;
        lm.position = transformPoint(inversePose(newIt->second), transformPoint(oldIt->second, lm.position));
        updateNormalAndDepth(map, id);
    }
}

LoopClosureResult correctLoop(Map &map, int keyframeId, const LoopGeometry &geometry, ExecutionMode mode) {
    LoopClosureResult result;
    result.detected = true;
    result.loopKeyframeId = geometry.candidateKeyframeId;
    result.matches = geometry.totalMatches;
    const int loopId = geometry.candidateKeyframeId;

    // Sec. VII-C: propagate the loop correction to the current keyframe's neighbours.
    updateCovisibility(map, keyframeId);
    std::vector<int> connected = covisibleKeyframes(map, keyframeId);
    connected.push_back(keyframeId);
    const std::set<int> connectedSet(connected.begin(), connected.end());

    std::map<int, Pose> corrected, nonCorrected;
    const Pose Twc = inversePose(map.keyframes.at(keyframeId).T_cw);
    for (int id : connected) {
        const Pose &Tiw = map.keyframes.at(id).T_cw;
        nonCorrected[id] = Tiw;
        corrected[id] = id == keyframeId ? geometry.correctedT_cw
                                         : composePose(composePose(Tiw, Twc), geometry.correctedT_cw);
    }

    std::vector<int> touched;
    for (const auto &[id, Tcorr] : corrected) {
        Keyframe &kf = map.keyframes.at(id);
        const Pose TwiCorrected = inversePose(Tcorr);
        const Pose &TiwOld = nonCorrected.at(id);
        for (int slot : kf.landmarkIds) {
            const int lmId = resolveLandmarkId(map, slot);
            if (lmId < 0) continue;
            Landmark &lm = map.landmarks.at(lmId);
            if (lm.correctedByKeyframeId == keyframeId) continue;
            lm.position = transformPoint(TwiCorrected, transformPoint(TiwOld, lm.position));
            lm.correctedByKeyframeId = keyframeId;
            lm.correctedReferenceId = id;
            touched.push_back(lmId);
        }
        kf.T_cw = Tcorr;
    }
    for (int lmId : touched)
        if (!map.landmarks.at(lmId).bad) updateNormalAndDepth(map, lmId);
    for (const auto &[id, Tcorr] : corrected)
        updateCovisibility(map, id);
    result.correctedKeyframes = static_cast<int>(corrected.size());

    // Loop fusion: matched loop points replace the current keyframe's duplicates.
    for (std::size_t i = 0; i < geometry.matchedLandmarkIds.size(); ++i) {
        const int loopLm = resolveLandmarkId(map, geometry.matchedLandmarkIds[i]);
        if (loopLm < 0) continue;
        const int existing = resolveLandmarkId(map, map.keyframes.at(keyframeId).landmarkIds[i]);
        if (existing >= 0) {
            if (existing != loopLm) {
                replaceLandmark(map, existing, loopLm);
                ++result.fusedLandmarks;
            }
        } else if (map.landmarks.at(loopLm).observations.count(keyframeId) == 0) {
            addObservation(map, loopLm, keyframeId, static_cast<int>(i));
            chooseRepresentativeDescriptor(map, loopLm);
            ++result.fusedLandmarks;
        }
    }
    // Project the loop-side points into all corrected keyframes and fuse.
    for (const auto &[id, Tcorr] : corrected)
        result.fusedLandmarks += fuseLandmarksIntoKeyframe(map, id, geometry.loopLandmarkIds,
                                                           map.config.loopFuseRadius, false, true);

    // New covisibility links that attach both sides of the loop.
    std::map<int, std::set<int>> loopConnections;
    for (int id : connected) {
        const std::vector<int> previousList = covisibleKeyframes(map, id);
        const std::set<int> previous(previousList.begin(), previousList.end());
        updateCovisibility(map, id);
        std::set<int> &links = loopConnections[id];
        for (const auto &[n, w] : map.keyframes.at(id).covisibilityWeights) {
            if (previous.count(n) == 0 && connectedSet.count(n) == 0) links.insert(n);
        }
    }

    // Sec. VII-D: essential graph optimization, then move the points with their keyframes.
    PoseGraph graph = buildEssentialGraph(map, keyframeId, loopId, nonCorrected, corrected, loopConnections);
    std::map<int, Pose> oldPoses, newPoses;
    for (std::size_t i = 0; i < graph.keyframeIds.size(); ++i)
        oldPoses[graph.keyframeIds[i]] = graph.poses[i];
    result.graph = optimizePoseGraph(graph, map.config.poseGraphIterations, mode);
    result.graphEdges = static_cast<int>(graph.edges.size());
    result.loopEdgesInGraph =
        static_cast<int>(std::count_if(graph.edges.begin(), graph.edges.end(),
                                       [](const GraphEdge &e) { return e.type == GraphEdgeType::Loop; }));
    for (std::size_t i = 0; i < graph.keyframeIds.size(); ++i) {
        newPoses[graph.keyframeIds[i]] = graph.poses[i];
        map.keyframes.at(graph.keyframeIds[i]).T_cw = graph.poses[i];
    }
    applyMapCorrection(map, oldPoses, newPoses, keyframeId);

    map.keyframes.at(keyframeId).loopEdgeIds.insert(loopId);
    map.keyframes.at(loopId).loopEdgeIds.insert(keyframeId);
    map.lastLoopKeyframeId = keyframeId;
    result.closed = true;
    return result;
}

LoopClosureResult runLoopClosing(Map &map, int keyframeId, ExecutionMode mode) {
    LoopClosureResult result;
    const std::vector<int> candidates = findLoopCandidates(map, keyframeId, mode);
    result.candidates = static_cast<int>(candidates.size());
    std::vector<LoopGeometry> geometries(candidates.size());
    detail::runPartitions(candidates.size(), detail::workerCount(candidates.size(), mode, 1),
                          [&](std::size_t begin, std::size_t end, std::size_t) {
                              for (std::size_t i = begin; i < end; ++i)
                                  geometries[i] = computeLoopGeometry(map, keyframeId, candidates[i], mode);
                          });
    for (const LoopGeometry &geometry : geometries) {
        if (!confirmLoopCandidate(map, keyframeId, geometry)) continue;
        result = correctLoop(map, keyframeId, geometry, mode);
        result.candidates = static_cast<int>(candidates.size());
        break;
    }
    return result;
}

} // namespace slam
