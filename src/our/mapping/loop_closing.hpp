#pragma once

// Loop closing (ORB-SLAM Sec. VII). Candidate scoring reuses person 1's keyframe similarity and
// matching, geometric verification reuses person 2's verifyLoopGeometry.

#include <map>
#include <limits>
#include <set>
#include <vector>

#include "map.hpp"
#include "pose_graph.hpp"
#include "teammate_interfaces.hpp"

namespace slam {

struct LoopGeometry {
    bool success = false;
    int candidateKeyframeId = -1;
    LoopGeometryResult geometry;         // person 2's hypothesis, refined using map reprojections
    Pose correctedT_cw;                  // current keyframe pose expressed in the loop side frame
    std::vector<int> matchedLandmarkIds; // per current keypoint: loop-side landmark or -1
    std::vector<int> loopLandmarkIds;    // landmarks of the candidate and its neighbours
    int inlierMatches = 0;
    int totalMatches = 0;
    double meanReprojectionChi2 = std::numeric_limits<double>::infinity();
};

struct LoopClosureResult {
    bool detected = false;
    bool closed = false;
    int loopKeyframeId = -1;
    int candidates = 0;
    int matches = 0;
    int fusedLandmarks = 0;
    int correctedKeyframes = 0;
    PoseGraphResult graph;
    int graphEdges = 0;
    int loopEdgesInGraph = 0;
};

// Sec. VII-A: score against covisible neighbours (theta_min = 30) gives s_min; older keyframes
// scoring at least s_min and not connected to the keyframe are candidates; a candidate is accepted
// after three consecutive group extensions (four detections). Updates map.consistentGroups.
std::vector<int> findLoopCandidates(Map &map, int keyframeId, ExecutionMode mode);

// Sec. VII-B: descriptor matches against the candidate, geometric verification, then guided
// search of the loop-side map points with the estimated transformation.
LoopGeometry computeLoopGeometry(const Map &map, int keyframeId, int candidateId, ExecutionMode mode);
// Fixed-scale form of the paper's bidirectional Sim(3) refinement. Matched IDs are
// candidate-side landmarks indexed by current keypoint; rejected entries become -1.
// The map is read-only and the pose maps candidate-camera coordinates to current.
int refineLoopTransform(const Map &map, int keyframeId, int candidateId,
                        std::vector<int> &matchedLandmarkIds, Pose &T_current_candidate, ExecutionMode mode);
// Checks the number, image distribution and alignment quality of the loop correspondences.
bool confirmLoopCandidate(const Map &map, int keyframeId, const LoopGeometry &geometry);

// Sec. VII-D: Essential Graph = spanning tree + covisibility edges with >= 100 shared points
// + loop edges. Measurements use the poses before correction, new loop links the corrected ones.
PoseGraph buildEssentialGraph(const Map &map, int currentId, int loopId,
                              const std::map<int, Pose> &nonCorrected, const std::map<int, Pose> &corrected,
                              const std::map<int, std::set<int>> &loopConnections);

// Moves every map point with the correction of its reference keyframe (Sec. VII-D).
void applyMapCorrection(Map &map, const std::map<int, Pose> &oldPoses, const std::map<int, Pose> &newPoses,
                        int currentKeyframeId);

// Sec. VII-C + VII-D: pose propagation, loop fusion, essential graph optimization.
LoopClosureResult correctLoop(Map &map, int keyframeId, const LoopGeometry &geometry, ExecutionMode mode);

// Full loop closing step for a new keyframe.
LoopClosureResult runLoopClosing(Map &map, int keyframeId, ExecutionMode mode);

} // namespace slam
