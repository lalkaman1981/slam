#pragma once

// Map, keyframes and map points following ORB-SLAM (Mur-Artal, Montiel, Tardos,
// IEEE T-RO 2015, arXiv:1502.00956), adapted to RGB-D input as in ORB-SLAM2.
// Section references (Sec. III-C, VI-B, ...) point to that paper.

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "slam_types.hpp"

namespace slam {

struct MapConfig {
    // ORB scale pyramid (Sec. III-A, 8 levels, factor 1.2).
    double scaleFactor = 1.2;
    int levels = 8;

    // New keyframe decision (Sec. V-E).
    int maxFramesAfterRelocalization = 20;
    int maxFramesBetweenKeyframes = 20;
    int minTrackedPoints = 50;
    double referenceTrackedRatio = 0.9;
    bool localMappingIdle = true; // serial pipeline: local mapping finishes before the next frame
    // RGB-D addition from ORB-SLAM2: insert when many close points are not tracked yet.
    bool useCloseDepthRule = true;
    double closeDepth = 3.2; // ORB-SLAM2 ThDepth * baseline for TUM RGB-D (metres)
    int closeTrackedLimit = 100;
    int closeUntrackedLimit = 70;

    // Map initialization and point creation from depth (ORB-SLAM2 RGB-D).
    int minInitialPoints = 500;
    int minNewPointsPerKeyframe = 100;

    // Covisibility graph (Sec. III-D) and essential graph (Sec. III-E).
    int covisibilityMinShared = 15;
    int essentialMinShared = 100;

    // Recent map point culling (Sec. VI-B).
    double minFoundRatio = 0.25;
    int minObservingKeyframes = 3;

    // New map point creation (Sec. VI-C).
    int triangulationNeighbors = 20;
    double minBaselineDepthRatio = 0.01;
    double maxParallaxCos = 0.9998;
    int descriptorThresholdLow = 50; // ORB-SLAM TH_LOW
    int fuseNeighbors = 20;
    int fuseSecondNeighbors = 5;
    double fuseRadius = 3.0;

    // Local bundle adjustment (Sec. VI-D, appendix).
    int baFirstIterations = 5;
    int baSecondIterations = 10;
    double chi2Mono = 5.991;   // 95% chi-square, 2 DoF
    double chi2Stereo = 7.815; // 95% chi-square, 3 DoF (RGB-D observation)
    bool useDepthInBundleAdjustment = true;

    // Local keyframe culling (Sec. VI-E).
    double redundantRatio = 0.9;
    int redundantObservers = 3;

    // Loop closing (Sec. VII).
    int keyframesBetweenLoops = 10;
    int loopNeighborMinShared = 30;
    int loopConsistency = 3;
    int loopMinKeyframeGap = 5; // exclude close temporal neighbours (team plan)
    int loopMinMatchesForGeometry = 20;
    int loopMinInliers = 20;
    int loopMinTotalMatches = 40;
    double loopProjectionRadius = 10.0;
    double loopFuseRadius = 4.0;
    int loopMinOccupiedCells = 4;   // distribution check on a 4x4 image grid (team plan)
    double loopMaxMeanError = 0.05; // metres, quality check (team plan)
    int poseGraphIterations = 20;

    // Keypoint grid used for area searches (ORB-SLAM FRAME_GRID_COLS / ROWS).
    int gridCols = 64;
    int gridRows = 48;
};

struct Keyframe {
    int id = -1;
    Frame frame;
    Pose T_cw;
    std::vector<int> landmarkIds; // per keypoint, -1 when unassociated

    std::map<int, int> covisibilityWeights; // keyframe id -> shared landmark count
    int parentId = -1;                      // spanning tree
    std::set<int> childIds;
    std::set<int> loopEdgeIds;

    bool bad = false;
    Pose T_cp;                          // pose relative to parent, kept when culled
    std::vector<std::vector<int>> grid; // keypoint indices per grid cell
};

struct Landmark {
    int id = -1;
    Vec3 position{};
    Vec3 normal{};            // mean viewing direction
    Descriptor descriptor{};  // representative ORB descriptor
    double minDistance = 0.0; // scale invariance limits
    double maxDistance = 0.0;
    std::map<int, int> observations; // keyframe id -> keypoint index
    int referenceKeyframeId = -1;
    int firstKeyframeId = -1;
    int visibleCount = 1;
    int foundCount = 1;
    bool bad = false;
    int replacedBy = -1;
    int correctedByKeyframeId = -1; // loop correction bookkeeping
    int correctedReferenceId = -1;
};

// Frame pose stored relative to its reference keyframe, so that it follows map corrections
// (ORB-SLAM saves the full-frame trajectory this way).
struct FrameRecord {
    double timestamp = 0.0;
    int referenceKeyframeId = -1;
    Pose T_cr;
};

struct ConsistentGroup {
    std::set<int> keyframeIds;
    int consistency = 0;
};

struct Map {
    CameraIntrinsics camera;
    MapConfig config;
    std::vector<double> scaleFactors;
    std::vector<double> levelSigma2;
    std::vector<double> invLevelSigma2;

    std::map<int, Keyframe> keyframes;
    std::map<int, Landmark> landmarks;
    int nextKeyframeId = 0;
    int nextLandmarkId = 0;
    int originKeyframeId = -1;

    std::vector<int> recentLandmarkIds;
    int lastKeyframeId = -1;
    int lastKeyframeFrameId = -1;
    int lastRelocalizationFrameId = -1;
    int lastLoopKeyframeId = 0;
    std::vector<ConsistentGroup> consistentGroups;
    std::vector<FrameRecord> frameRecords;
};

struct LocalMap {
    std::vector<int> keyframeIds;
    std::vector<int> landmarkIds;
    int referenceKeyframeId = -1;
};

struct VisibleLandmark {
    int landmarkId = -1;
    double u = 0.0;
    double v = 0.0;
    int predictedLevel = 0;
    double viewCos = 0.0;
};

Map createMap(const CameraIntrinsics &camera, const MapConfig &config = MapConfig{});

// ---- Functions from the team plan (person 3) ----
int initializeMap(Map &map, const Frame &firstFrame);
bool shouldInsertKeyframe(const Frame &frame, const TrackingResult &tracking, const Map &map);
int insertKeyframe(Map &map, const Frame &frame, const Pose &T_cw);
LocalMap selectLocalMap(const Map &map, const std::vector<int> &trackedLandmarkIds);
std::vector<VisibleLandmark> findVisibleLandmarks(const Map &map, const LocalMap &localMap,
                                                  const Pose &predictedPose, const CameraIntrinsics &camera,
                                                  ExecutionMode mode);
int associateObservations(Map &map, int keyframeId, const TrackingResult &tracking);
std::vector<int> createLandmarks(Map &map, int keyframeId, const std::vector<int> &unmatchedFeatures);
void chooseRepresentativeDescriptor(Map &map, int landmarkId);
int cullLandmarks(Map &map, int currentKeyframeId);
void updateCovisibility(Map &map, int keyframeId);

// ---- Local mapping steps from Sec. VI that the plan table folds into updateMap ----
int triangulateNewLandmarks(Map &map, int keyframeId);
int fuseLandmarksInNeighbors(Map &map, int keyframeId);
int cullKeyframes(Map &map, int currentKeyframeId);

// ---- Map maintenance helpers ----
std::vector<int> unmatchedFeatures(const Map &map, int keyframeId);
int resolveLandmarkId(const Map &map, int landmarkId); // follows replacements, -1 if bad
void addObservation(Map &map, int landmarkId, int keyframeId, int keypointIdx);
void eraseObservation(Map &map, int landmarkId, int keyframeId);
void setLandmarkBad(Map &map, int landmarkId);
void replaceLandmark(Map &map, int oldId, int newId);
void setKeyframeBad(Map &map, int keyframeId);
void updateNormalAndDepth(Map &map, int landmarkId);
int predictScaleLevel(const Map &map, const Landmark &landmark, double distance);
std::vector<int> covisibleKeyframes(const Map &map, int keyframeId, int minWeight = 0);
std::vector<int> bestCovisibleKeyframes(const Map &map, int keyframeId, std::size_t n);
int trackedLandmarks(const Map &map, int keyframeId, int minObservations);
std::vector<int> featuresInArea(const Map &map, const Keyframe &keyframe, double u, double v, double radius,
                                int minLevel, int maxLevel);
// Projection-based fusion (ORB-SLAM ORBmatcher::Fuse). Returns the number of fused points.
int fuseLandmarksIntoKeyframe(Map &map, int keyframeId, const std::vector<int> &landmarkIds,
                              double radiusFactor, bool checkReprojection, bool preferIncoming);
// Matches landmarks to unassociated keypoints by projection with the given pose
// (ORB-SLAM ORBmatcher::SearchByProjection). Fills matchedLandmarks, returns new matches.
int searchByProjection(const Map &map, const Keyframe &keyframe, const Pose &T_cw,
                       const std::vector<int> &landmarkIds, std::vector<int> &matchedLandmarks,
                       double radiusFactor);

} // namespace slam
