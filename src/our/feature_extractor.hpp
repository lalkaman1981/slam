#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
#include <opencv2/core.hpp>

constexpr size_t INIT_CORNERS = 128;
constexpr size_t RESERVED_CANDIDATES_INIT = 100;
constexpr int FAST_CORNER_THRESHOLD = 20;
constexpr int FAST_ARC_LENGTH = 12;
// Scale factor between pyramid levels. Defaults below were chosen with our_tune (tune_params.cpp)
// on TUM freiburg1_xyz and freiburg1_room, judged with depth and ground-truth poses.
constexpr double PYRAMID_SCALE = 1.3;
// Candidates stay this far from the border (orientation radius / BRIEF patch half-size is 15).
constexpr int PATCH_BORDER = 16;
constexpr int DESC_SIZE = 4;

using pyramid_t = std::vector<cv::Mat>;
using pyramid_level_t = cv::Mat;
using level_corner_t = std::vector<std::pair<int, int>>;
using pyramid_corners_t = std::vector<level_corner_t>;
using candidates_scores_t = std::vector<float>;
using level_scores_t = std::vector<candidates_scores_t>;
using brief_desc_t = std::array<long long, 4>;

struct Candidate {
    float score;
    std::pair<int, int> coords;
    int level;
};

struct ORB_Descriptor {
    Candidate candidate;
    brief_desc_t descriptor;
    float orientation;
};

using orb_matches_t = std::unordered_map<size_t, size_t>;

// Tunable parameters of the feature extractor.
struct OrbParams {
    int fast_threshold = FAST_CORNER_THRESHOLD;  // intensity difference for the FAST segment test
    int fast_arc = FAST_ARC_LENGTH;              // contiguous circle pixels required
    int levels = 8;                              // pyramid levels
    double scale = PYRAMID_SCALE;                // scale factor between levels
    int top_k = 16000;                           // corners kept per image before grid selection (more than needed on purpose)
    int grid_side = 8;                           // grid cell size (level-0 pixels)
    int max_in_cell = 3;                         // keypoints kept per grid cell
    float harris_k = 0.05f;
    int harris_radius = 3;
    int blur_ksize = 7;                          // smoothing before BRIEF (odd)
    double blur_sigma = 2.0;
};

// Tunable parameters of the matcher.
struct MatchParams {
    int dist_thrs = 80;      // maximum Hamming distance of the best match
    double alpha = 0.85;     // ratio test: best < alpha * second best
    bool angle_check = false; // rotation consistency filter (cost more correct matches than it removed outliers)
    int angle_bins = 30;
    int angle_top_bins = 3;
    double angle_min_ratio = 0.1;
};


struct MapCandidate {
    ORB_Descriptor descriptor;
    float predicted_x;
    float predicted_y;
    int predicted_level = -1;
};

pyramid_t build_pyramid(const cv::Mat& gray,
    int level_num = 8, double scale_factor = PYRAMID_SCALE);

pyramid_corners_t detect_fast_corners(
    const pyramid_t& pyramid,
    int threshold = FAST_CORNER_THRESHOLD,
    int arc_length = FAST_ARC_LENGTH);

candidates_scores_t evaluate_level(
    const level_corner_t& level_candidates,
    const pyramid_level_t& level,
    const int radius = 3,
    const float k = 0.05f);

level_scores_t get_corners_scores(
    const pyramid_corners_t& pyramid_candidates,
    const pyramid_t& pyramid,
    int radius = 3,
    float k = 0.05f);

int scale_k(
    int curr_level,
    int top_k = 100,
    int level_num = 8,
    float scale = static_cast<float>(PYRAMID_SCALE));

std::pair<int, int> scale_coords(std::pair<int, int> coords, double rescale = PYRAMID_SCALE);

std::vector<Candidate> select_keypoints_by_grid(
    const std::vector<Candidate>& candidates,
    int grid_side = 8,
    int max_in_cell = 3,
    double scale = PYRAMID_SCALE);

std::vector<Candidate> create_valid_candidates(
    const pyramid_corners_t& pyramid_candidates,
    const level_scores_t& pyramid_scores,
    int top_k = 100,
    float scale = static_cast<float>(PYRAMID_SCALE),
    int grid_side = 8,
    int max_in_cell = 3);

float compute_orientation(
    const cv::Mat& image,
    const std::pair<int, int>& point);

std::vector<float> get_orientations(
    const pyramid_t& pyramid,
    const std::vector<Candidate>& candidates);

// 256 sample pairs (x1, y1, x2, y2), relative to the patch centre, for a given patch size.
const std::vector<std::array<int, 4>>& get_brief_pattern(int kernel_size);

brief_desc_t compute_brief_descriptor(
    pyramid_level_t level,
    std::pair<int, int> point,
    int kernel_size = 31);

std::vector<brief_desc_t> get_descriptors(
    const pyramid_t& pyramid,
    const std::vector<Candidate>& candidates);

std::vector<ORB_Descriptor> create_orb_descs(
    const std::vector<Candidate>& candidates,
    const std::vector<brief_desc_t>& descriptors,
    const std::vector<float>& orientations);

std::vector<ORB_Descriptor> extractFeatures(const cv::Mat& gray, const OrbParams& params = {});

// Convenience overload: default parameters with only the keypoint budget changed.
// Note that top_k is the number of corners kept before the grid selection, so the
// number of returned features is smaller.
std::vector<ORB_Descriptor> extractFeatures(const cv::Mat& gray, int top_k);

int hamming_distance(const brief_desc_t& desc1, const brief_desc_t& desc2);

bool filter_descriptor_matches(
    int dist1, int dist2,
    int dist_thrs = 80,
    double alpha = 0.85);

std::optional<size_t> find_two_nearest(
    const ORB_Descriptor& query,
    const std::vector<ORB_Descriptor>& candidates,
    int dist_thrs = 80,
    double alpha = 0.85);

orb_matches_t get_matches(
    const std::vector<ORB_Descriptor>& orb_descriptors_query,
    const std::vector<ORB_Descriptor>& orb_descriptors_key,
    int dist_thrs = 80,
    double alpha = 0.85);

orb_matches_t mutual_check(
    const orb_matches_t& matches12,
    const orb_matches_t& matches21);


orb_matches_t filter_matches_by_angle(
    const orb_matches_t& matches,
    const std::vector<float>& angles_query,
    const std::vector<float>& angles_key,
    int bins = 30,
    int top_bins = 3,
    double min_ratio = 0.1);

orb_matches_t filter_matches_by_angle(
    const orb_matches_t& matches,
    const std::vector<ORB_Descriptor>& features_query,
    const std::vector<ORB_Descriptor>& features_key,
    int bins = 30,
    int top_bins = 3,
    double min_ratio = 0.1);


orb_matches_t match_frames(
    const std::vector<ORB_Descriptor>& features1,
    const std::vector<ORB_Descriptor>& features2,
    bool angle_check = false);

orb_matches_t match_frames(
    const std::vector<ORB_Descriptor>& features1,
    const std::vector<ORB_Descriptor>& features2,
    const MatchParams& params);

orb_matches_t match_frames(
    const cv::Mat& gray1,
    const cv::Mat& gray2,
    bool angle_check = false);

double score_keyframe_similarity(
    const std::vector<ORB_Descriptor>& a,
    const std::vector<ORB_Descriptor>& b);

orb_matches_t match_frame2map(
    const std::vector<ORB_Descriptor>& frame,
    const std::vector<MapCandidate>& candidates,
    float search_radius = 15.0f,
    int level_tolerance = 1,
    double scale = PYRAMID_SCALE,
    int dist_thrs = 100,
    double alpha = 0.9,
    bool angle_check = false);
