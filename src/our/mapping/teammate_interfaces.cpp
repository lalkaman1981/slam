#include "teammate_interfaces.hpp"
#include "integration_adapters.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace slam {

int hammingDistance(const Descriptor &a, const Descriptor &b) {
    return ::hamming_distance(toTrackingDescriptor(a), toTrackingDescriptor(b));
}

std::vector<Match> matchFrames(const Frame &a, const Frame &b, ExecutionMode /*mode*/) {
    // Upstream modules currently expose their serial implementation. Mapping mode
    // controls mapping kernels; it does not claim to vectorize the teammates' code.
    const auto matches = ::match_frames(toTrackingFeatures(a), toTrackingFeatures(b));
    std::vector<Match> result;
    result.reserve(matches.size());
    for (const auto &[query, train] : matches)
        result.push_back({static_cast<int>(query), static_cast<int>(train),
                          hammingDistance(a.descriptors[query], b.descriptors[train])});
    std::sort(result.begin(), result.end(),
              [](const Match &x, const Match &y) { return x.queryIdx < y.queryIdx; });
    return result;
}

double scoreKeyframeSimilarity(const Frame &a, const Frame &b) {
    return ::score_keyframe_similarity(toTrackingFeatures(a), toTrackingFeatures(b));
}

Vec3 backProject(double u, double v, double depth, const CameraIntrinsics &camera) {
    const cv::Point3d point = ::backProject({u, v}, depth, toCameraMatrix(camera));
    return makeVec3(point.x, point.y, point.z);
}

bool project(const Vec3 &pointCamera, const CameraIntrinsics &camera, double &u, double &v) {
    const auto pixel = ::project({pointCamera[0], pointCamera[1], pointCamera[2]}, toCameraMatrix(camera));
    if (!pixel) return false;
    u = pixel->x;
    v = pixel->y;
    return std::isfinite(u) && std::isfinite(v) && u >= 0.0 && u < camera.width && v >= 0.0 &&
           v < camera.height;
}

Pose estimateRigidTransform(const std::vector<Vec3> &source, const std::vector<Vec3> &target) {
    if (source.size() != target.size())
        throw std::invalid_argument("rigid alignment needs equally sized point arrays");
    std::vector<PointPair3D> pairs;
    pairs.reserve(source.size());
    for (std::size_t i = 0; i < source.size(); ++i)
        pairs.push_back(
            {{source[i][0], source[i][1], source[i][2]}, {target[i][0], target[i][1], target[i][2]}, i, i});
    const auto pose = ::estimateRigidTransform(pairs);
    if (!pose) throw std::invalid_argument("rigid alignment needs three nondegenerate point pairs");
    return toMappingPose(*pose);
}

LoopGeometryResult verifyLoopGeometry(const Frame &current, const Frame &candidate,
                                      const std::vector<Match> &matches, const CameraIntrinsics &camera,
                                      ExecutionMode /*mode*/) {
    LoopGeometryResult result;
    result.inliers.assign(matches.size(), false);
    std::vector<PointPair3D> pairs;
    std::vector<std::size_t> originalIndices;
    const cv::Matx33d K = toCameraMatrix(camera);
    for (std::size_t i = 0; i < matches.size(); ++i) {
        const Match &match = matches[i];
        if (match.queryIdx < 0 || match.trainIdx < 0) continue;
        const auto query = static_cast<std::size_t>(match.queryIdx);
        const auto train = static_cast<std::size_t>(match.trainIdx);
        if (query >= current.keypoints.size() || query >= current.depths.size() ||
            train >= candidate.keypoints.size() || train >= candidate.depths.size())
            continue;
        const double currentDepth = current.depths[query];
        const double candidateDepth = candidate.depths[train];
        if (!std::isfinite(currentDepth) || !std::isfinite(candidateDepth) || currentDepth < MIN_DEPTH ||
            currentDepth > MAX_DEPTH || candidateDepth < MIN_DEPTH || candidateDepth > MAX_DEPTH)
            continue;
        const Keypoint &currentPoint = current.keypoints[query];
        const Keypoint &candidatePoint = candidate.keypoints[train];
        // Mapping matches are current -> candidate. RANSAC maps candidate -> current.
        // Keep per-feature depths instead of rasterizing a synthetic depth image:
        // distinct features can round onto the same pixel.
        pairs.push_back({::backProject({candidatePoint.u, candidatePoint.v}, candidateDepth, K),
                         ::backProject({currentPoint.u, currentPoint.v}, currentDepth, K), train, query});
        originalIndices.push_back(i);
    }
    const auto geometry = ::estimatePoseRansac(pairs);
    if (!geometry) return result;
    result.T_current_candidate = toMappingPose(geometry->T_ba);
    const auto errors = ::computeAlignmentErrors(pairs, geometry->T_ba);
    double totalError = 0.0;
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        if (!geometry->inliers[i]) continue;
        result.inliers[originalIndices[i]] = true;
        ++result.inlierCount;
        totalError += errors[i];
    }
    result.meanError = result.inlierCount > 0 ? totalError / result.inlierCount : 0.0;
    result.success = result.inlierCount >= static_cast<int>(MIN_INLIERS);
    return result;
}

} // namespace slam
