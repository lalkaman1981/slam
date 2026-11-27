#include "rgbd_pipeline.hpp"
#include "processor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <stdexcept>

namespace slam {
namespace {
using Clock = std::chrono::steady_clock;
double secondsSince(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
PipelineConfig checkedConfig(const PipelineConfig &config) {
    const auto &c = config.camera;
    if (!std::isfinite(c.fx) || !std::isfinite(c.fy) || !std::isfinite(c.cx) || !std::isfinite(c.cy) ||
        c.fx <= 0 || c.fy <= 0 || c.width < 64 || c.height < 64 ||
        config.maxFeatures < static_cast<int>(MIN_INLIERS) || config.mapping.minInitialPoints < 3 ||
        !std::isfinite(c.bf) || c.bf <= 0)
        throw std::invalid_argument("Invalid camera or pipeline configuration");
    if (config.mapping.levels != 8 || std::abs(config.mapping.scaleFactor - 1.2) > 1e-12)
        throw std::invalid_argument("Upstream features use eight pyramid levels with scale 1.2");
    return config;
}
} // namespace

RgbdPipeline::RgbdPipeline(const PipelineConfig &config)
    : config_(checkedConfig(config)), map_(createMap(config_.camera, config_.mapping)) {}

Pose RgbdPipeline::recordedPose(size_t index) const {
    const auto &record = map_.frameRecords.at(index);
    Pose relative = record.T_cr;
    int id = record.referenceKeyframeId;
    while (map_.keyframes.at(id).bad) {
        relative = composePose(relative, map_.keyframes.at(id).T_cp);
        id = map_.keyframes.at(id).parentId;
    }
    return composePose(relative, map_.keyframes.at(id).T_cw);
}

TrackingResult RgbdPipeline::associateMap(const ::Frame &frame, const Pose &initialPose,
                                          const std::vector<int> &seedIds, bool relocalized) {
    TrackingResult tracking;
    tracking.ok = true; // incoming pose has already passed upstream depth RANSAC
    tracking.T_cw = initialPose;
    tracking.relocalized = relocalized;
    tracking.landmarkIds.assign(frame.features.size(), -1);
    LocalMap local = selectLocalMap(map_, seedIds);
    if (local.keyframeIds.empty())
        local = selectLocalMap(map_, map_.keyframes.at(map_.lastKeyframeId).landmarkIds);
    tracking.referenceKeyframeId = local.referenceKeyframeId;
    const auto visible = findVisibleLandmarks(map_, local, initialPose, map_.camera, config_.mode);
    std::vector<MapCandidate> candidates;
    std::vector<cv::Point3d> points;
    std::vector<int> ids;
    for (const auto &view : visible) {
        const auto &lm = map_.landmarks.at(view.landmarkId);
        ORB_Descriptor descriptor{};
        descriptor.descriptor = toTrackingDescriptor(lm.descriptor);
        descriptor.candidate.level = view.predictedLevel;
        // A representative point's angle is not a current-camera orientation. Disable
        // angle filtering below; geometry supplies the final association test.
        candidates.push_back(
            {descriptor, static_cast<float>(view.u), static_cast<float>(view.v), view.predictedLevel});
        points.emplace_back(lm.position[0], lm.position[1], lm.position[2]);
        ids.push_back(view.landmarkId);
        tracking.visibleLandmarkIds.push_back(view.landmarkId);
    }
    const auto projectedMatches =
        ::match_frame2map(frame.features, candidates, 15.0f, 1, 1.2, 80, 0.8, false);
    // Carry verified frame-to-frame associations forward. Rematching these points
    // would unnecessarily lose observations to the projection matcher's ratio test.
    orb_matches_t matches;
    std::set<int> usedLandmarks;
    for (size_t feature = 0; feature < seedIds.size(); ++feature) {
        const int id = resolveLandmarkId(map_, seedIds[feature]);
        if (id < 0 || !usedLandmarks.insert(id).second) continue;
        auto it = std::find(ids.begin(), ids.end(), id);
        if (it == ids.end()) {
            const auto &point = map_.landmarks.at(id).position;
            matches.emplace(feature, points.size());
            points.emplace_back(point[0], point[1], point[2]);
            ids.push_back(id);
        } else
            matches.emplace(feature, static_cast<size_t>(it - ids.begin()));
    }
    for (const auto &[feature, point] : projectedMatches) {
        if (matches.count(feature) == 0 && usedLandmarks.insert(ids.at(point)).second)
            matches.emplace(feature, point);
    }
    const auto K = toCameraMatrix(map_.camera);
    const auto observations = ::buildMapObservations(frame, points, matches);
    const auto refined = ::refinePose(observations, toTrackingPose(initialPose), K);
    if (refined && refined->numInliers >= MIN_INLIERS) tracking.T_cw = toMappingPose(refined->T_cw);

    // Only geometrically verified map matches may become persistent observations.
    const auto T = toTrackingPose(tracking.T_cw);
    for (const auto &[feature, point] : matches) {
        const auto &f = frame.features.at(feature);
        const double scale = std::pow(1.2, f.candidate.level);
        const cv::Point2d pixel(f.candidate.coords.first * scale, f.candidate.coords.second * scale);
        const auto pc = ::transformPoints(std::span<const cv::Point3d>(&points.at(point), 1), T).front();
        const auto projected = ::project(pc, K);
        if (!projected || cv::norm(*projected - pixel) > HUBER_DELTA * scale) continue;
        const auto measuredDepth = ::readValidDepth(frame, {cvRound(pixel.x), cvRound(pixel.y)});
        if (measuredDepth && std::abs(pc.z - *measuredDepth) > std::max(0.05, 0.02 * *measuredDepth))
            continue;
        tracking.landmarkIds.at(feature) = ids.at(point);
    }
    return tracking;
}

PipelineResult RgbdPipeline::process(const cv::Mat &image, const cv::Mat &depth, double timestamp) {
    if (image.empty() || image.depth() != CV_8U ||
        (image.channels() != 1 && image.channels() != 3 && image.channels() != 4) ||
        depth.type() != CV_16UC1 || image.size() != depth.size() || image.cols != map_.camera.width ||
        image.rows != map_.camera.height)
        throw std::invalid_argument(
            "Expected calibrated-size 8-bit RGB/gray and registered 16-bit depth images");
    if (!std::isfinite(timestamp) || (lastTimestamp_ && timestamp <= *lastTimestamp_))
        throw std::invalid_argument("Frame timestamps must be finite and strictly increasing");
    lastTimestamp_ = timestamp;
    PipelineResult result;
    auto start = Clock::now();
    ::Frame frame{::extractFeatures(::prepare_image(image), config_.maxFeatures), depth.clone()};
    result.features = frame.features.size();
    const auto mapped = toMappingFrame(frame, nextFrameId_++, timestamp, map_.camera);
    result.featureSeconds = secondsSince(start);

    TrackingResult tracking;
    if (!map_.keyframes.empty()) {
        start = Clock::now();
        TrackResult pose;
        std::vector<int> seedIds(frame.features.size(), -1);
        if (previous_ && !wasLost_) {
            const auto matches = ::match_frames(previous_->features, frame.features);
            result.matches = matches.size();
            pose = ::trackFrame(frame, *previous_, toTrackingPose(recordedPose(previousRecord_)), matches, {},
                                {}, toCameraMatrix(map_.camera));
            for (const auto &[ref, current] : pose.inlierMatches) {
                if (ref < previousLandmarks_.size()) seedIds.at(current) = previousLandmarks_[ref];
            }
        }
        if (pose.status == TrackingStatus::LOST) {
            std::vector<std::pair<double, int>> scores;
            for (const auto &[id, cached] : keyframeImages_) {
                if (!map_.keyframes.at(id).bad)
                    scores.emplace_back(::score_keyframe_similarity(frame.features, cached.features), id);
            }
            std::sort(scores.begin(), scores.end(), [](const auto &a, const auto &b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
            for (size_t i = 0; i < std::min(scores.size(), config_.relocalizationCandidates); ++i) {
                const int id = scores[i].second;
                const auto &cached = keyframeImages_.at(id);
                const auto matches = ::match_frames(cached.features, frame.features);
                const RelocalizationCandidate candidate{cached, toTrackingPose(map_.keyframes.at(id).T_cw),
                                                        matches};
                const auto recovered = ::relocalizeFrame(frame, {&candidate, 1}, toCameraMatrix(map_.camera));
                if (recovered.status != TrackingStatus::OK ||
                    recovered.inlierMatches.size() <= pose.inlierMatches.size())
                    continue;
                pose = recovered;
                result.matches = matches.size();
                result.relocalized = true;
                std::fill(seedIds.begin(), seedIds.end(), -1);
                for (const auto &[ref, current] : recovered.inlierMatches)
                    seedIds.at(current) = map_.keyframes.at(id).landmarkIds.at(ref);
            }
        }
        result.inliers = pose.inlierMatches.size();
        if (pose.status == TrackingStatus::OK)
            tracking = associateMap(frame, toMappingPose(pose.T_cw), seedIds, result.relocalized);
        result.trackingSeconds = secondsSince(start);
    }

    start = Clock::now();
    result.mapping = updateMap(map_, mapped, tracking, config_.mode);
    result.mappingSeconds = secondsSince(start);
    result.initialized = result.mapping.initialized;
    result.tracked = result.initialized || tracking.ok;
    if (result.tracked) {
        previous_ = std::move(frame);
        previousRecord_ = map_.frameRecords.size() - 1;
        result.T_cw = recordedPose(previousRecord_);
        previousLandmarks_ = result.mapping.keyframeInserted
                                 ? map_.keyframes.at(result.mapping.keyframeId).landmarkIds
                                 : tracking.landmarkIds;
        if (result.mapping.keyframeInserted) keyframeImages_[result.mapping.keyframeId] = *previous_;
        for (auto it = keyframeImages_.begin(); it != keyframeImages_.end();) {
            if (map_.keyframes.at(it->first).bad)
                it = keyframeImages_.erase(it);
            else
                ++it;
        }
    }
    wasLost_ = !result.tracked;
    return result;
}

} // namespace slam
