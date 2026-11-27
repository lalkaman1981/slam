#pragma once

// updateMap: the per-frame entry point of module 3. Runs the ORB-SLAM Local Mapping steps
// (Sec. VI A-E) for a new keyframe and then Loop Closing (Sec. VII) on it, sequentially.

#include <string>

#include "bundle_adjustment.hpp"
#include "loop_closing.hpp"
#include "map.hpp"

namespace slam {

struct MapUpdateResult {
    bool initialized = false;
    bool keyframeInserted = false;
    int keyframeId = -1;
    int associated = 0;
    int createdFromDepth = 0;
    int triangulated = 0;
    int fused = 0;
    int culledLandmarks = 0;
    int culledKeyframes = 0;
    BAResult localBA;
    LoopClosureResult loop;
};

MapUpdateResult updateMap(Map &map, const Frame &frame, const TrackingResult &tracking, ExecutionMode mode);

// Map points as an ASCII PLY point cloud.
bool exportMap(const Map &map, const std::string &path);
// Every tracked frame in TUM format (timestamp tx ty tz qx qy qz qw, camera-to-world),
// reconstructed through the reference keyframes so it includes all map corrections.
bool exportTrajectory(const Map &map, const std::string &path);
// Keyframe trajectory in TUM format.
bool exportKeyframeTrajectory(const Map &map, const std::string &path);

} // namespace slam
