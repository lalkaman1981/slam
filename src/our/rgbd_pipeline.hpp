#pragma once

#include "pose_estimation.hpp"
#include "mapping/integration_adapters.hpp"
#include "mapping/map_pipeline.hpp"

#include <map>
#include <optional>

namespace slam {

struct PipelineConfig {
    CameraIntrinsics camera;
    // The upstream grid filter often returns far fewer than maxFeatures. Use
    // thresholds suitable for that extractor; the mapping-only demo retains its
    // original dense synthetic defaults.
    MapConfig mapping = [] {
        MapConfig config;
        config.minInitialPoints = 60;
        config.minTrackedPoints = 15;
        return config;
    }();
    ExecutionMode mode = ExecutionMode::Serial;
    int maxFeatures = 1000;
    size_t relocalizationCandidates = 5;
};

struct PipelineResult {
    bool initialized = false;
    bool tracked = false;
    bool relocalized = false;
    size_t features = 0;
    size_t matches = 0;
    size_t inliers = 0;
    Pose T_cw;
    MapUpdateResult mapping;
    double featureSeconds = 0.0;
    double trackingSeconds = 0.0;
    double mappingSeconds = 0.0;
};

// RGB-D images are registered to the calibrated pinhole camera; depth is CV_16UC1,
// 5000 units/metre, as consumed by person 2. Input frames must have increasing times.
// The mode selects person 3's implementation. Upstream modules remain serial.
class RgbdPipeline {
  public:
    explicit RgbdPipeline(const PipelineConfig &config = {});
    PipelineResult process(const cv::Mat &image, const cv::Mat &depth, double timestamp);
    const Map &map() const { return map_; }

  private:
    Pose recordedPose(size_t index) const;
    TrackingResult associateMap(const ::Frame &frame, const Pose &initialPose,
                                const std::vector<int> &seedIds, bool relocalized);

    PipelineConfig config_;
    Map map_;
    int nextFrameId_ = 0;
    std::optional<double> lastTimestamp_;
    std::optional<::Frame> previous_;
    size_t previousRecord_ = 0;
    std::vector<int> previousLandmarks_;
    std::map<int, ::Frame> keyframeImages_;
    bool wasLost_ = false;
};

} // namespace slam
