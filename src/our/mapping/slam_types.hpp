#pragma once

// Compact mapping storage. integration_adapters.hpp converts the upstream
// feature and tracking structures without modifying either teammate's interface.

#include <array>
#include <cstdint>
#include <vector>

#include "se3.hpp"

namespace slam {

// Mapping implementations share one interface; upstream feature/tracking remain serial.
enum class ExecutionMode { Serial, SerialSimd, Parallel };

// 256-bit ORB descriptor.
using Descriptor = std::array<std::uint8_t, 32>;

struct Keypoint {
    float u = 0.0f; // pixel coordinates in the original (level 0) image
    float v = 0.0f;
    int octave = 0;     // pyramid level the corner was detected at
    float angle = 0.0f; // orientation in degrees, as in ORB
    float response = 0.0f;
};

struct Match {
    int queryIdx = -1;
    int trainIdx = -1;
    int distance = 0;
};

struct CameraIntrinsics {
    double fx = 525.0;
    double fy = 525.0;
    double cx = 319.5;
    double cy = 239.5;
    int width = 640;
    int height = 480;
    // ORB-SLAM2 RGB-D "virtual stereo" baseline times fx (Camera.bf). Used to turn a depth
    // measurement into a right-image coordinate u_R = u - bf / depth for bundle adjustment.
    double bf = 40.0;
};

// Converted feature output with depth sampled by person 2's readValidDepth.
// depths[i] <= 0 means no valid depth for keypoint i.
struct Frame {
    int id = -1;
    double timestamp = 0.0;
    std::vector<Keypoint> keypoints;
    std::vector<Descriptor> descriptors;
    std::vector<float> depths;
};

// Verified tracking output plus persistent map associations prepared by the pipeline.
struct TrackingResult {
    bool ok = false;
    Pose T_cw;                           // estimated camera pose, world -> camera
    std::vector<int> landmarkIds;        // per keypoint: matched inlier landmark id or -1
    std::vector<int> visibleLandmarkIds; // local-map landmarks predicted visible (findVisibleLandmarks)
    int referenceKeyframeId = -1;        // keyframe sharing most landmarks with the frame
    bool relocalized = false;            // pose came from relocalization in this frame
};

} // namespace slam
