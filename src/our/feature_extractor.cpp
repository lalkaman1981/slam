#include "feature_extractor.hpp"

#include <algorithm>
#include <bit>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <numeric>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int circle[16][2] = {
    { 0,-3},{ 1,-3},{ 2,-2},{ 3,-1},{ 3, 0},{ 3, 1},{ 2, 2},{ 1, 3},
    { 0, 3},{-1, 3},{-2, 2},{-3, 1},{-3, 0},{-3,-1},{-2,-2},{-1,-3}
};

const cv::Mat sobel_x = (cv::Mat_<float>(3, 3) <<
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
);

const cv::Mat sobel_y = (cv::Mat_<float>(3, 3) <<
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1
);

} // namespace

pyramid_t build_pyramid(const cv::Mat& gray,
    int level_num, double scale_factor) {
    if (gray.empty()) throw std::invalid_argument("gray image is empty");
    if (level_num < 1) throw std::invalid_argument("level_num must be bigger than 0");
    if (scale_factor <= 1.0) throw std::invalid_argument("scale_factor must be > 1");

    pyramid_t pyramid;
    pyramid.reserve(level_num);
    pyramid.push_back(gray);
    cv::Size new_size = cv::Size(gray.cols, gray.rows);

    for (int i = 1; i < level_num; ++i) {
        const double s = 1.0 / std::pow(scale_factor, i);
        int h = static_cast<int>(std::lround(gray.rows * s));
        int w = static_cast<int>(std::lround(gray.cols * s));
        if (h < 1 || w < 1) break;

        cv::Mat level;
        new_size.width = w;
        new_size.height = h;
        cv::resize(gray, level, new_size, 0, 0, cv::INTER_AREA);
        pyramid.push_back(std::move(level));
    }

    return pyramid;
}

static bool is_fast_corner(const cv::Mat& img, int x, int y) {
    const int v = img.at<uchar>(y, x);
    int state[16];
    for (int k = 0; k < 16; ++k) {
        const int p = img.at<uchar>(y + circle[k][1], x + circle[k][0]);
        state[k] = p > v + FAST_CORNER_THRESHOLD ? 1 : (p < v - FAST_CORNER_THRESHOLD ? -1 : 0);
    }
    for (int sign : {1, -1}) {
        int run = 0;
        for (int k = 0; k < 16 + FAST_ARC_LENGTH - 1; ++k) {
            if (state[k % 16] == sign) {
                if (++run >= FAST_ARC_LENGTH) return true;
            } else {
                run = 0;
            }
        }
    }
    return false;
}

pyramid_corners_t detect_fast_corners(const pyramid_t& pyramid) {
    pyramid_corners_t pyramid_candidates;
    pyramid_candidates.reserve(pyramid.size());

    for (const auto& level : pyramid) {
        level_corner_t level_vec;
        level_vec.reserve(INIT_CORNERS);

        for (int y = PATCH_BORDER; y < level.rows - PATCH_BORDER; ++y) {
            for (int x = PATCH_BORDER; x < level.cols - PATCH_BORDER; ++x) {
                if (is_fast_corner(level, x, y)) level_vec.emplace_back(x, y);
            }
        }

        pyramid_candidates.push_back(std::move(level_vec));
    }

    return pyramid_candidates;
}

static cv::Mat get_derivative(const cv::Mat& level, const cv::Mat& sobel_axis) {
    cv::Mat level32, I_axis;
    level.convertTo(level32, CV_32F);
    cv::filter2D(level32, I_axis, CV_32F, sobel_axis);
    return I_axis;
}

candidates_scores_t evaluate_level(
    const level_corner_t& level_candidates,
    const pyramid_level_t& level,
    const int radius,
    const float k) {

    cv::Mat Ix = get_derivative(level, sobel_x);
    cv::Mat Iy = get_derivative(level, sobel_y);

    const cv::Size win(2 * radius + 1, 2 * radius + 1);
    cv::Mat sxx, syy, sxy;
    cv::boxFilter(Ix.mul(Ix), sxx, CV_32F, win, cv::Point(-1, -1), false);
    cv::boxFilter(Iy.mul(Iy), syy, CV_32F, win, cv::Point(-1, -1), false);
    cv::boxFilter(Ix.mul(Iy), sxy, CV_32F, win, cv::Point(-1, -1), false);

    candidates_scores_t candidates_scores;
    candidates_scores.reserve(level_candidates.size());

    for (const auto& [x, y] : level_candidates) {
        const float A = sxx.at<float>(y, x);
        const float B = sxy.at<float>(y, x);
        const float C = syy.at<float>(y, x);
        candidates_scores.push_back(A * C - B * B - k * (A + C) * (A + C));
    }

    return candidates_scores;
}

level_scores_t get_corners_scores(
    const pyramid_corners_t& pyramid_candidates,
    const pyramid_t& pyramid) {

    level_scores_t pyramid_scores;
    pyramid_scores.reserve(pyramid.size());
    for (size_t i = 0; i < pyramid.size(); ++i) {
        pyramid_scores.push_back(evaluate_level(pyramid_candidates[i], pyramid[i]));
    }
    return pyramid_scores;
}

int scale_k(
    int curr_level,
    int top_k,
    int level_num,
    float scale) {
    float q = 1 / scale;
    return static_cast<int>(top_k * (1 - q) * std::pow(q, curr_level) / (1 - std::pow(q, level_num)));
}

std::pair<int, int> scale_coords(std::pair<int, int> coords, double rescale) {
    return {static_cast<int>(std::lround(coords.first * rescale)),
            static_cast<int>(std::lround(coords.second * rescale))};
}

std::vector<Candidate> select_keypoints_by_grid(
    const std::vector<Candidate>& candidates,
    int grid_side,
    int max_in_cell,
    double scale) {

    std::map<std::pair<int, int>, int> level_map;
    std::vector<Candidate> res_candidates;
    res_candidates.reserve(RESERVED_CANDIDATES_INIT);

    for (const auto& candidate : candidates) {
        auto [sx, sy] = scale_coords(candidate.coords, std::pow(scale, candidate.level));
        const std::pair<int, int> cell(sx / grid_side, sy / grid_side);
        int& count = level_map[cell];
        if (count >= max_in_cell) continue;
        ++count;
        res_candidates.push_back(candidate);
    }

    return res_candidates;
}

std::vector<Candidate> create_valid_candidates(
    const pyramid_corners_t& pyramid_candidates,
    const level_scores_t& pyramid_scores,
    int top_k,
    float scale) {
    const int level_num = static_cast<int>(pyramid_scores.size());

    std::vector<Candidate> candidates;

    for (int level_idx = 0; level_idx < level_num; ++level_idx) {
        const auto& scores = pyramid_scores[level_idx];
        const auto& corners = pyramid_candidates[level_idx];

        const size_t n = std::min(
            scores.size(),
            static_cast<size_t>(std::max(0, scale_k(level_idx, top_k, level_num, scale))));

        std::vector<size_t> order(scores.size());
        std::iota(order.begin(), order.end(), size_t{0});
        std::partial_sort(order.begin(), order.begin() + n, order.end(),
            [&](size_t a, size_t b) { return scores[a] > scores[b]; });

        for (size_t i = 0; i < n; ++i) {
            candidates.push_back(Candidate{scores[order[i]], corners[order[i]], level_idx});
        }
    }

    return select_keypoints_by_grid(candidates, 10, 3, scale);
}

float compute_orientation(
    const cv::Mat& image,
    const std::pair<int, int>& point) {
    constexpr int radius = 15;

    int m10 = 0;
    int m01 = 0;

    for (int v = -radius; v <= radius; ++v) {
        const int umax = static_cast<int>(std::sqrt(radius * radius - v * v));
        const uchar* row = image.ptr<uchar>(point.second + v);

        for (int u = -umax; u <= umax; ++u) {
            const int intensity = row[point.first + u];
            m10 += u * intensity;
            m01 += v * intensity;
        }
    }

    return std::atan2(static_cast<float>(m01), static_cast<float>(m10));
}

std::vector<float> get_orientations(
    const pyramid_t& pyramid,
    const std::vector<Candidate>& candidates) {

    std::vector<float> orientations;
    orientations.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        orientations.push_back(compute_orientation(pyramid[candidate.level], candidate.coords));
    }
    return orientations;
}

static const std::vector<std::array<int, 4>>& get_brief_pattern(int kernel_size) {
    static std::map<int, std::vector<std::array<int, 4>>> cache;
    auto it = cache.find(kernel_size);
    if (it != cache.end()) return it->second;

    const int half = kernel_size / 2;
    const double sigma = kernel_size / 5.0;
    cv::RNG rng(0x5EED);

    auto sample = [&]() {
        return std::clamp(static_cast<int>(std::lround(rng.gaussian(sigma))), -half, half);
    };

    std::vector<std::array<int, 4>> pattern;
    pattern.reserve(256);
    for (int i = 0; i < 256; ++i) {
        int x1 = sample();
        int y1 = sample();
        int x2 = sample();
        int y2 = sample();
        pattern.push_back({x1, y1, x2, y2});
    }
    return cache.emplace(kernel_size, std::move(pattern)).first->second;
}

brief_desc_t compute_brief_descriptor(
    pyramid_level_t level,
    std::pair<int, int> point,
    int kernel_size) {

    if (level.empty() || level.type() != CV_8UC1)
        throw std::invalid_argument("level must be a non-empty CV_8UC1 image");
    if (kernel_size < 1 || kernel_size % 2 == 0)
        throw std::invalid_argument("kernel_size must be odd and positive");

    const int half = kernel_size / 2;
    const auto [x, y] = point;
    if (x < half || y < half || x >= level.cols - half || y >= level.rows - half)
        throw std::out_of_range("point is too close to the image border");

    const auto& pattern = get_brief_pattern(kernel_size);
    std::array<uint64_t, 4> bits{};

    for (size_t i = 0; i < pattern.size(); ++i) {
        const auto& p = pattern[i];
        const uchar a = level.at<uchar>(y + p[1], x + p[0]);
        const uchar b = level.at<uchar>(y + p[3], x + p[2]);
        if (a < b) bits[i / 64] |= (uint64_t{1} << (i % 64));
    }

    brief_desc_t descriptor;
    for (int i = 0; i < 4; ++i) descriptor[i] = static_cast<long long>(bits[i]);
    return descriptor;
}

std::vector<brief_desc_t> get_descriptors(
    const pyramid_t& pyramid,
    const std::vector<Candidate>& candidates) {

    std::vector<brief_desc_t> descriptors;
    descriptors.reserve(candidates.size());
    for (const auto& candidate : candidates) {
        descriptors.push_back(compute_brief_descriptor(pyramid[candidate.level], candidate.coords));
    }
    return descriptors;
}

std::vector<ORB_Descriptor> create_orb_descs(
    const std::vector<Candidate>& candidates,
    const std::vector<brief_desc_t>& descriptors,
    const std::vector<float>& orientations) {

    std::vector<ORB_Descriptor> orb_descriptors;
    orb_descriptors.reserve(candidates.size());
    for (size_t i = 0; i < candidates.size(); ++i) {
        orb_descriptors.push_back(ORB_Descriptor{candidates[i], descriptors[i], orientations[i]});
    }
    return orb_descriptors;
}

std::vector<ORB_Descriptor> extractFeatures(const cv::Mat& gray, int top_k) {
    if (gray.empty() || gray.type() != CV_8UC1)
        throw std::invalid_argument("gray must be a non-empty CV_8UC1 image");

    pyramid_t pyramid = build_pyramid(gray);
    pyramid_corners_t pyramid_candidates = detect_fast_corners(pyramid);
    level_scores_t pyramid_scores = get_corners_scores(pyramid_candidates, pyramid);
    std::vector<Candidate> candidates = create_valid_candidates(pyramid_candidates, pyramid_scores, top_k);
    std::vector<float> orientations = get_orientations(pyramid, candidates);

    pyramid_t smoothed(pyramid.size());
    for (size_t i = 0; i < pyramid.size(); ++i) {
        cv::GaussianBlur(pyramid[i], smoothed[i], cv::Size(7, 7), 2);
    }
    std::vector<brief_desc_t> descriptors = get_descriptors(smoothed, candidates);

    return create_orb_descs(candidates, descriptors, orientations);
}

int hamming_distance(const brief_desc_t& desc1, const brief_desc_t& desc2) {
    int dist = 0;
    for (int i = 0; i < DESC_SIZE; ++i) dist += std::popcount(static_cast<uint64_t>(desc1[i] ^ desc2[i]));
    return dist;
}

bool filter_descriptor_matches(
    int dist1, int dist2,
    int dist_thrs,
    double alpha) {
    bool ratio_test = static_cast<double>(dist1) < alpha * dist2;
    bool dist_test = dist1 < dist_thrs;
    return ratio_test && dist_test;
}

std::optional<size_t> find_two_nearest(
    const ORB_Descriptor& query,
    const std::vector<ORB_Descriptor>& candidates) {

    int dist1 = INT_MAX, dist2 = INT_MAX;
    size_t idx1 = 0;

    for (size_t i = 0; i < candidates.size(); ++i) {
        int dist = hamming_distance(query.descriptor, candidates[i].descriptor);
        if (dist < dist1) {
            dist2 = dist1;
            dist1 = dist;
            idx1 = i;
        } else if (dist < dist2) {
            dist2 = dist;
        }
    }

    if (dist2 == INT_MAX) return std::nullopt;
    if (!filter_descriptor_matches(dist1, dist2)) return std::nullopt;
    return idx1;
}

orb_matches_t get_matches(
    const std::vector<ORB_Descriptor>& orb_descriptors_query,
    const std::vector<ORB_Descriptor>& orb_descriptors_key) {
    orb_matches_t orb_matches;
    orb_matches.reserve(orb_descriptors_query.size());
    for (size_t q = 0; q < orb_descriptors_query.size(); ++q) {
        if (auto best = find_two_nearest(orb_descriptors_query[q], orb_descriptors_key)) {
            orb_matches.emplace(q, *best);
        }
    }
    return orb_matches;
}

orb_matches_t mutual_check(
    const orb_matches_t& matches12,
    const orb_matches_t& matches21) {
    orb_matches_t result;
    result.reserve(matches12.size());
    for (const auto& [q, c] : matches12) {
        auto it = matches21.find(c);
        if (it != matches21.end() && it->second == q) result.emplace(q, c);
    }
    return result;
}

orb_matches_t filter_matches_by_angle(
    const orb_matches_t& matches,
    const std::vector<float>& angles_query,
    const std::vector<float>& angles_key,
    int bins,
    int top_bins,
    double min_ratio) {

    if (bins < 1 || top_bins < 1) throw std::invalid_argument("bins and top_bins must be positive");
    if (matches.empty()) return matches;

    constexpr float two_pi = 2.0f * static_cast<float>(M_PI);
    auto bin_of = [&](size_t q, size_t k) {
        float d = std::fmod(angles_key[k] - angles_query[q], two_pi);
        if (d < 0) d += two_pi;
        return std::min(bins - 1, static_cast<int>(d / two_pi * static_cast<float>(bins)));
    };

    std::vector<int> histogram(bins, 0);
    for (const auto& [q, k] : matches) ++histogram[bin_of(q, k)];

    std::vector<int> order(bins);
    std::iota(order.begin(), order.end(), 0);
    const int n_top = std::min(top_bins, bins);
    std::partial_sort(order.begin(), order.begin() + n_top, order.end(),
        [&](int a, int b) { return histogram[a] > histogram[b] || (histogram[a] == histogram[b] && a < b); });

    std::vector<bool> keep_bin(bins, false);
    const int max_count = histogram[order[0]];
    for (int i = 0; i < n_top; ++i) {
        if (i == 0 || histogram[order[i]] >= min_ratio * max_count) keep_bin[order[i]] = true;
    }

    orb_matches_t result;
    result.reserve(matches.size());
    for (const auto& [q, k] : matches) {
        if (keep_bin[bin_of(q, k)]) result.emplace(q, k);
    }
    return result;
}

orb_matches_t filter_matches_by_angle(
    const orb_matches_t& matches,
    const std::vector<ORB_Descriptor>& features_query,
    const std::vector<ORB_Descriptor>& features_key,
    int bins,
    int top_bins,
    double min_ratio) {

    std::vector<float> angles_query, angles_key;
    angles_query.reserve(features_query.size());
    angles_key.reserve(features_key.size());
    for (const auto& f : features_query) angles_query.push_back(f.orientation);
    for (const auto& f : features_key) angles_key.push_back(f.orientation);
    return filter_matches_by_angle(matches, angles_query, angles_key, bins, top_bins, min_ratio);
}

orb_matches_t match_frames(
    const std::vector<ORB_Descriptor>& features1,
    const std::vector<ORB_Descriptor>& features2,
    bool angle_check) {

    orb_matches_t matches12 = get_matches(features1, features2);
    orb_matches_t matches21 = get_matches(features2, features1);

    orb_matches_t mutual = mutual_check(matches12, matches21);
    if (angle_check) mutual = filter_matches_by_angle(mutual, features1, features2);
    return mutual;
}

orb_matches_t match_frames(
    const cv::Mat& gray1,
    const cv::Mat& gray2,
    bool angle_check) {

    return match_frames(extractFeatures(gray1), extractFeatures(gray2), angle_check);
}

double score_keyframe_similarity(
    const std::vector<ORB_Descriptor>& a,
    const std::vector<ORB_Descriptor>& b) {

    if (a.empty() || b.empty()) return 0.0;
    const orb_matches_t mutual = match_frames(a, b);
    return static_cast<double>(mutual.size()) / static_cast<double>(std::min(a.size(), b.size()));
}

orb_matches_t match_frame2map(
    const std::vector<ORB_Descriptor>& frame,
    const std::vector<MapCandidate>& candidates,
    float search_radius,
    int level_tolerance,
    double scale,
    int dist_thrs,
    double alpha,
    bool angle_check) {

    std::vector<std::pair<float, float>> positions;
    positions.reserve(frame.size());
    for (const auto& f : frame) {
        auto [x, y] = scale_coords(f.candidate.coords, std::pow(scale, f.candidate.level));
        positions.emplace_back(static_cast<float>(x), static_cast<float>(y));
    }

    const float r2 = search_radius * search_radius;
    std::unordered_map<size_t, std::pair<size_t, int>> best_for_frame;

    for (size_t m = 0; m < candidates.size(); ++m) {
        const auto& cand = candidates[m];
        int dist1 = INT_MAX, dist2 = INT_MAX;
        size_t idx1 = 0;

        for (size_t f = 0; f < frame.size(); ++f) {
            const float dx = positions[f].first - cand.predicted_x;
            const float dy = positions[f].second - cand.predicted_y;
            if (dx * dx + dy * dy > r2) continue;
            if (cand.predicted_level >= 0 &&
                std::abs(frame[f].candidate.level - cand.predicted_level) > level_tolerance) continue;

            const int dist = hamming_distance(cand.descriptor.descriptor, frame[f].descriptor);
            if (dist < dist1) {
                dist2 = dist1;
                dist1 = dist;
                idx1 = f;
            } else if (dist < dist2) {
                dist2 = dist;
            }
        }

        if (dist1 == INT_MAX || !filter_descriptor_matches(dist1, dist2, dist_thrs, alpha)) continue;

        auto it = best_for_frame.find(idx1);
        if (it == best_for_frame.end() || dist1 < it->second.second) {
            best_for_frame[idx1] = {m, dist1};
        }
    }

    orb_matches_t matches;
    matches.reserve(best_for_frame.size());
    for (const auto& [f, cm] : best_for_frame) matches.emplace(f, cm.first);

    if (angle_check) {
        std::vector<float> angles_frame, angles_map;
        angles_frame.reserve(frame.size());
        angles_map.reserve(candidates.size());
        for (const auto& f : frame) angles_frame.push_back(f.orientation);
        for (const auto& m : candidates) angles_map.push_back(m.descriptor.orientation);
        matches = filter_matches_by_angle(matches, angles_frame, angles_map);
    }
    return matches;
}
