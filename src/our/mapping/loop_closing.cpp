// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: http://www.viva64.com

#include "loop_closing.hpp"
#include "bundle_adjustment.hpp"
#include "execution.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace slam {

namespace {

constexpr int kDistributionGrid = 4;
constexpr std::size_t kGroupNeighbors = 10;
constexpr double kGroupRetainRatio = 0.75;
constexpr double kLoopReprojectionChi2 = 10.0;

int covisibilityWeight(const Map &map, int a, int b) {
    const auto &weights = map.keyframes.at(a).covisibilityWeights;
    const auto it = weights.find(b);
    return it == weights.end() ? 0 : it->second;
}

// Sec. VII-B: search in both cameras and accept only reciprocal map-point
// matches. The transformation has unit scale because this project initializes
// from metric depth.
void guidedLoopMatches(const Map &map, const Keyframe &current, const Keyframe &candidate,
                       const Pose &T_current_candidate, std::vector<int> &matches) {
    std::vector<bool> usedCurrent(current.landmarkIds.size()), usedCandidate(candidate.landmarkIds.size());
    for (std::size_t i = 0; i < matches.size(); ++i) {
        const int id = resolveLandmarkId(map, matches[i]);
        if (id < 0) continue;
        usedCurrent[i] = true;
        const auto observation = map.landmarks.at(id).observations.find(candidate.id);
        if (observation != map.landmarks.at(id).observations.end())
            usedCandidate[static_cast<std::size_t>(observation->second)] = true;
    }
    auto search = [&](const Keyframe &source, const Keyframe &target, const Pose &T_target_source,
                      const std::vector<bool> &usedSource, const std::vector<bool> &usedTarget) {
        std::vector<int> found(source.landmarkIds.size(), -1);
        for (std::size_t i = 0; i < source.landmarkIds.size(); ++i) {
            const int id = resolveLandmarkId(map, source.landmarkIds[i]);
            if (id < 0 || usedSource[i]) continue;
            const Landmark &landmark = map.landmarks.at(id);
            const Vec3 point =
                transformPoint(T_target_source, transformPoint(source.T_cw, landmark.position));
            double u, v;
            if (!project(point, map.camera, u, v)) continue;
            const double distance = norm(point);
            if (distance < 0.8 * landmark.minDistance || distance > 1.2 * landmark.maxDistance) continue;
            const int level = predictScaleLevel(map, landmark, distance);
            int bestDistance = 101;
            for (int index : featuresInArea(map, target, u, v, 7.5 * map.scaleFactors[level],
                                            std::max(0, level - 1), level)) {
                const auto j = static_cast<std::size_t>(index);
                if (usedTarget[j] || resolveLandmarkId(map, target.landmarkIds[j]) < 0) continue;
                const int distance = hammingDistance(landmark.descriptor, target.frame.descriptors[j]);
                if (distance < bestDistance) {
                    bestDistance = distance;
                    found[i] = index;
                }
            }
        }
        return found;
    };
    const auto forward =
        search(current, candidate, inversePose(T_current_candidate), usedCurrent, usedCandidate);
    const auto backward = search(candidate, current, T_current_candidate, usedCandidate, usedCurrent);
    for (std::size_t i = 0; i < forward.size(); ++i) {
        const int j = forward[i];
        if (j >= 0 && backward[static_cast<std::size_t>(j)] == static_cast<int>(i))
            matches[i] = resolveLandmarkId(map, candidate.landmarkIds[static_cast<std::size_t>(j)]);
    }
}

// Fixed-scale form of the paper's OptimizeSim3 objective: both image
// reprojections, octave information and Huber loss. Reuse our batched BA
// kernels and native 6x6 solver. Neither map points nor keyframes are mutated
// while evaluating a candidate.
int refineLoopTransformImpl(const Map &map, const Keyframe &current, const Keyframe &candidate,
                            std::vector<int> &matches, Pose &T_current_candidate, ExecutionMode mode,
                            double &meanReprojectionChi2) {
    meanReprojectionChi2 = std::numeric_limits<double>::infinity();
    BAProblem problem;
    problem.camera = map.camera;
    problem.poses = {T_current_candidate, inversePose(T_current_candidate)};
    problem.points.reserve(2 * matches.size());
    problem.observations.reserve(2 * matches.size());
    std::vector<std::size_t> featureIndices;
    featureIndices.reserve(matches.size());
    for (std::size_t i = 0; i < matches.size(); ++i) {
        const int currentId = resolveLandmarkId(map, current.landmarkIds[i]);
        const int candidateId = resolveLandmarkId(map, matches[i]);
        if (currentId < 0 || candidateId < 0) {
            matches[i] = -1;
            continue;
        }
        const Landmark &landmark = map.landmarks.at(candidateId);
        const auto observed = landmark.observations.find(candidate.id);
        if (observed == landmark.observations.end()) {
            matches[i] = -1;
            continue;
        }
        const auto &a = current.frame.keypoints[i];
        const auto &b = candidate.frame.keypoints[static_cast<std::size_t>(observed->second)];
        const auto point = problem.points.size();
        problem.points.push_back(transformPoint(candidate.T_cw, landmark.position));
        problem.points.push_back(transformPoint(current.T_cw, map.landmarks.at(currentId).position));
        problem.observations.push_back(
            {0, point, makeVec3(a.u, a.v, 0), false, map.invLevelSigma2[a.octave]});
        problem.observations.push_back(
            {1, point + 1, makeVec3(b.u, b.v, 0), false, map.invLevelSigma2[b.octave]});
        featureIndices.push_back(i);
    }
    if (featureIndices.size() < 10) return 0;
    std::vector<ObservationResidual> residuals;
    auto evaluate = [&](bool jacobians) {
        computeObservationResiduals(problem, residuals, jacobians, mode);
        double cost = 0.0;
        for (std::size_t i = 0; i < residuals.size(); ++i) {
            if (!problem.observations[i].active) continue;
            const auto &r = residuals[i];
            if (!r.inFront || !std::isfinite(r.chi2)) return std::numeric_limits<double>::infinity();
            cost += r.chi2 <= kLoopReprojectionChi2
                        ? r.chi2
                        : 2.0 * std::sqrt(kLoopReprojectionChi2 * r.chi2) - kLoopReprojectionChi2;
        }
        return cost;
    };
    auto optimize = [&](int iterations) {
        double lambda = 1e-5;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const double cost = evaluate(true);
            if (!std::isfinite(cost)) break;
            Mat6 H;
            Vec6 gradient;
            for (std::size_t i = 0; i < residuals.size(); ++i) {
                if (!problem.observations[i].active) continue;
                const auto &r = residuals[i];
                Mat36 J = r.jacobianPose;
                if (i % 2 != 0) {
                    // T21' = T21 * exp(-delta): derivative in camera 1 coordinates.
                    Mat36 inverseIncrement;
                    setBlock(inverseIncrement, 0, 0, skew(problem.points[i]));
                    setBlock(inverseIncrement, 0, 3, -Mat3::identity());
                    J = detail::multiply(r.jacobianPoint, inverseIncrement, mode);
                }
                const double weight =
                    problem.observations[i].information *
                    (r.chi2 <= kLoopReprojectionChi2 ? 1.0 : std::sqrt(kLoopReprojectionChi2 / r.chi2));
                H += weight * detail::multiply(J.transpose(), J, mode);
                gradient += weight * (J.transpose() * r.error);
            }
            DenseMatrix damped(6, 6);
            std::vector<double> rhs(6), increment;
            for (std::size_t r = 0; r < 6; ++r) {
                rhs[r] = -gradient[r];
                for (std::size_t c = 0; c < 6; ++c) damped(r, c) = H(r, c);
                damped(r, r) += lambda * std::max(1.0, H(r, r));
            }
            if (!solveCholesky(damped, rhs, increment, mode)) break;
            Vec6 delta;
            std::copy(increment.begin(), increment.end(), delta.data.begin());
            const Pose before = problem.poses[0];
            problem.poses[0] = composePose(expSE3(delta), before);
            problem.poses[1] = inversePose(problem.poses[0]);
            if (evaluate(false) < cost)
                lambda = std::max(1e-12, lambda * 0.1);
            else {
                problem.poses[0] = before;
                problem.poses[1] = inversePose(before);
                lambda *= 10.0;
            }
            if (norm(delta) < 1e-9) break;
        }
    };
    auto rejectOutliers = [&] {
        evaluate(false);
        int inliers = 0;
        for (std::size_t i = 0; i < featureIndices.size(); ++i) {
            const auto &a = residuals[2 * i], &b = residuals[2 * i + 1];
            const bool accepted = problem.observations[2 * i].active && a.inFront && b.inFront &&
                                  std::isfinite(a.chi2) && std::isfinite(b.chi2) &&
                                  a.chi2 <= kLoopReprojectionChi2 && b.chi2 <= kLoopReprojectionChi2;
            problem.observations[2 * i].active = problem.observations[2 * i + 1].active = accepted;
            if (accepted)
                ++inliers;
            else
                matches[featureIndices[i]] = -1;
        }
        return inliers;
    };
    optimize(5);
    const int firstInliers = rejectOutliers();
    if (firstInliers < 10) return 0;
    optimize(firstInliers < static_cast<int>(featureIndices.size()) ? 10 : 5);
    const int inliers = rejectOutliers();
    if (inliers > 0) {
        meanReprojectionChi2 = 0.0;
        for (std::size_t i = 0; i < residuals.size(); ++i)
            if (problem.observations[i].active) meanReprojectionChi2 += residuals[i].chi2;
        meanReprojectionChi2 /= 2 * inliers;
    }
    T_current_candidate = problem.poses[0];
    return inliers;
}

} // namespace

int refineLoopTransform(const Map &map, int keyframeId, int candidateId, std::vector<int> &matchedLandmarkIds,
                        Pose &T_current_candidate, ExecutionMode mode) {
    const auto &current = map.keyframes.at(keyframeId);
    if (matchedLandmarkIds.size() != current.frame.keypoints.size())
        throw std::invalid_argument("loop matches must have one slot per current keypoint");
    double meanReprojectionChi2;
    return refineLoopTransformImpl(map, current, map.keyframes.at(candidateId), matchedLandmarkIds,
                                   T_current_candidate, mode, meanReprojectionChi2);
}

std::vector<int> findLoopCandidates(Map &map, int keyframeId, ExecutionMode mode) {
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
        for (int n : connectedKeyframes(map, candidateId)) group.insert(n);
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

    // Both sides must refer to persistent map points, as in the paper's
    // Sim3Solver. Sensor depth is used when points are created, not to gate an
    // existing loop.
    std::vector<Match> matches;
    std::vector<Vec3> currentPoints, candidatePoints;
    for (const Match &m : matchFrames(current.frame, candidate.frame, mode)) {
        const int currentId =
            resolveLandmarkId(map, current.landmarkIds[static_cast<std::size_t>(m.queryIdx)]);
        const int candidateId =
            resolveLandmarkId(map, candidate.landmarkIds[static_cast<std::size_t>(m.trainIdx)]);
        if (currentId < 0 || candidateId < 0) continue;
        matches.push_back(m);
        currentPoints.push_back(transformPoint(current.T_cw, map.landmarks.at(currentId).position));
        candidatePoints.push_back(transformPoint(candidate.T_cw, map.landmarks.at(candidateId).position));
    }
    if (static_cast<int>(matches.size()) < map.config.loopMinMatchesForGeometry) return result;

    result.geometry = verifyLoopGeometry(currentPoints, candidatePoints, mode);
    if (!result.geometry.success) return result;

    result.matchedLandmarkIds.assign(current.frame.keypoints.size(), -1);
    for (std::size_t m = 0; m < matches.size(); ++m) {
        if (!result.geometry.inliers[m]) continue;
        result.matchedLandmarkIds[static_cast<std::size_t>(matches[m].queryIdx)] =
            resolveLandmarkId(map, candidate.landmarkIds[static_cast<std::size_t>(matches[m].trainIdx)]);
    }

    guidedLoopMatches(map, current, candidate, result.geometry.T_current_candidate,
                      result.matchedLandmarkIds);
    result.inlierMatches = refineLoopTransformImpl(map, current, candidate, result.matchedLandmarkIds,
                                                  result.geometry.T_current_candidate, mode,
                                                  result.meanReprojectionChi2);
    if (result.inlierMatches < map.config.loopMinInliers) return result;
    result.geometry.meanError = 0.0;
    for (std::size_t i = 0; i < result.matchedLandmarkIds.size(); ++i) {
        const int candidatePoint = result.matchedLandmarkIds[i];
        if (candidatePoint < 0) continue;
        const int currentPoint = resolveLandmarkId(map, current.landmarkIds[i]);
        const Vec3 a = transformPoint(candidate.T_cw, map.landmarks.at(candidatePoint).position);
        const Vec3 b = transformPoint(current.T_cw, map.landmarks.at(currentPoint).position);
        result.geometry.meanError += norm(transformPoint(result.geometry.T_current_candidate, a) - b);
    }
    result.geometry.meanError /= result.inlierMatches;
    for (std::size_t i = 0; i < matches.size(); ++i)
        result.geometry.inliers[i] = result.matchedLandmarkIds[static_cast<std::size_t>(matches[i].queryIdx)] ==
            resolveLandmarkId(map, candidate.landmarkIds[static_cast<std::size_t>(matches[i].trainIdx)]);
    result.geometry.inlierCount = static_cast<int>(
        std::count(result.geometry.inliers.begin(), result.geometry.inliers.end(), true));
    result.correctedT_cw = composePose(result.geometry.T_current_candidate, candidate.T_cw);

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
    if (!std::isfinite(geometry.meanReprojectionChi2) ||
        geometry.meanReprojectionChi2 > kLoopReprojectionChi2) return false;

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
    auto addCorrectedLink = [&](int i, int j) {
        if (!hasNode(i) || !hasNode(j) || i == j || !inserted.insert(std::minmax(i, j)).second) return;
        addLoopEdge(graph, j, i, composePose(initial(j), inversePose(initial(i))));
    };
    for (const auto &[i, links] : loopConnections) {
        if (!hasNode(i)) continue;
        for (int j : links) {
            if (!hasNode(j)) continue;
            if ((i != currentId || j != loopId) && covisibilityWeight(map, i, j) < minShared) continue;
            addCorrectedLink(i, j);
        }
    }
    // The geometrically verified loop is a constraint even when fusion introduced
    // no new shared-point edge (for example, all corresponding points were
    // already fused).
    addCorrectedLink(currentId, loopId);

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

    // Snapshot every old neighborhood before fusion changes reciprocal
    // connections.
    std::map<int, std::set<int>> previousConnections;
    for (int id : connected) {
        const auto previous = connectedKeyframes(map, id);
        previousConnections[id] = {previous.begin(), previous.end()};
    }

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
    for (int id : connected) updateCovisibility(map, id);
    for (int id : connected) {
        std::set<int> &links = loopConnections[id];
        for (int neighbor : connectedKeyframes(map, id))
            if (!previousConnections.at(id).contains(neighbor) && !connectedSet.contains(neighbor))
                links.insert(neighbor);
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
    const auto workers = detail::workerCount(candidates.size(), mode, 1);
    // Parallelize candidates or their residuals, avoiding nested worker groups.
    const auto candidateMode = workers > 1 ? ExecutionMode::Serial : mode;
    detail::runPartitions(candidates.size(), workers,
                          [&](std::size_t begin, std::size_t end, std::size_t) {
                              for (std::size_t i = begin; i < end; ++i)
                                  geometries[i] = computeLoopGeometry(map, keyframeId, candidates[i], candidateMode);
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
