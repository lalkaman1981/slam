#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "options_parser.h"
#include "time_measurements.cpp"

#include "execution_mode.hpp"
#include "pose_estimation.hpp"

#ifdef BENCH_OPENCV
#include <cfloat>
#include <opencv2/calib3d.hpp>
#endif

// every operation is built once per variant (see bench/CMakeLists.txt):
// <op>_serial and <op>_simd run our code in that execution mode, <op>_opencv runs the
// OpenCV counterpart. BENCHMARK.md lists how the methods differ.
#if defined(BENCH_SIMD)
constexpr slam::ExecutionMode bench_mode = slam::ExecutionMode::SerialSimd;
#else
constexpr slam::ExecutionMode bench_mode = slam::ExecutionMode::Serial;
#endif

// results go here after the measurement, so the compiler cannot drop the measured work
inline long long sink = 0;

inline long long micro(double value) { return static_cast<long long>(value * 1e6); }

#ifdef BENCH_OPENCV
// cv::estimateAffine3D fits a 12-dof affine; its 3x3 part is projected to the closest
// rotation (SVD) and becomes the starting pose of cv::solvePnPRefineLM
inline void rigid_from_affine(const cv::Mat& affine, cv::Mat& rvec, cv::Mat& tvec) {
    const cv::SVD svd(affine(cv::Rect(0, 0, 3, 3)));
    cv::Mat R = svd.u * svd.vt;
    if (cv::determinant(R) < 0) R = svd.u * cv::Mat::diag((cv::Mat_<double>(3, 1) << 1, 1, -1)) * svd.vt;
    cv::Rodrigues(R, rvec);
    tvec = affine.col(3).clone();
}

// the same iteration cap as refinePose
inline cv::TermCriteria refine_criteria() {
    return {cv::TermCriteria::COUNT + cv::TermCriteria::EPS, REFINE_MAX_ITERATIONS, FLT_EPSILON};
}
#endif

// a synthetic RGB-D sequence: manifest line "side fx fy cx cy depth_scale frames",
// then one line "gray.png depth.png" per frame, paths relative to the manifest
struct Sequence {
    int side = 0;
    cv::Matx33d K;
    double depth_scale = 0.0;
    std::vector<cv::Mat> gray;
    std::vector<cv::Mat> depth;
};

inline Sequence read_sequence(const std::string& path) {
    std::ifstream file(path);
    Sequence sequence;
    double fx, fy, cx, cy;
    int frames;
    if (!(file >> sequence.side >> fx >> fy >> cx >> cy >> sequence.depth_scale >> frames))
        throw std::runtime_error("bad sequence manifest " + path);
    sequence.K = cv::Matx33d(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    const auto folder = std::filesystem::path(path).parent_path();
    std::string gray, depth;
    for (int i = 0; i < frames && file >> gray >> depth; ++i) {
        sequence.gray.push_back(cv::imread((folder / gray).string(), cv::IMREAD_GRAYSCALE));
        sequence.depth.push_back(cv::imread((folder / depth).string(), cv::IMREAD_ANYDEPTH));
        if (sequence.gray.back().empty() || sequence.depth.back().type() != CV_16UC1)
            throw std::runtime_error("cannot read frame " + std::to_string(i) + " of " + path);
    }
    if (static_cast<int>(sequence.gray.size()) != frames) throw std::runtime_error("missing frames in " + path);
    return sequence;
}

// pose input: "fx fy cx cy", then "N" and N lines "ax ay az bx by bz" (3D-3D pairs),
// then "M" and M lines "X Y Z u v" (3D-2D observations of the second camera)
struct PoseInput {
    cv::Matx33d K;
    std::vector<PointPair3D> pairs;
    std::vector<MapObservation> observations;
};

inline PoseInput read_pose_input(const std::string& path) {
    std::ifstream file(path);
    PoseInput input;
    double fx, fy, cx, cy;
    size_t n, m;
    if (!(file >> fx >> fy >> cx >> cy >> n)) throw std::runtime_error("bad pose input " + path);
    input.K = cv::Matx33d(fx, 0, cx, 0, fy, cy, 0, 0, 1);
    for (size_t i = 0; i < n; ++i) {
        PointPair3D p;
        file >> p.a.x >> p.a.y >> p.a.z >> p.b.x >> p.b.y >> p.b.z;
        p.idxA = p.idxB = i;
        input.pairs.push_back(p);
    }
    file >> m;
    for (size_t i = 0; i < m; ++i) {
        MapObservation o;
        file >> o.pointWorld.x >> o.pointWorld.y >> o.pointWorld.z >> o.pixel.x >> o.pixel.y;
        input.observations.push_back(o);
    }
    if (!file) throw std::runtime_error("truncated pose input " + path);
    return input;
}

// linear system, binary: int64 n, then n*n doubles (row-major A), then n doubles (b)
struct LinearSystem {
    size_t n = 0;
    std::vector<double> A;
    std::vector<double> b;
};

inline LinearSystem read_system(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    std::int64_t n = 0;
    file.read(reinterpret_cast<char*>(&n), sizeof n);
    if (!file || n <= 0) throw std::runtime_error("bad system file " + path);
    LinearSystem system;
    system.n = static_cast<size_t>(n);
    system.A.resize(system.n * system.n);
    system.b.resize(system.n);
    file.read(reinterpret_cast<char*>(system.A.data()), static_cast<std::streamsize>(system.A.size() * sizeof(double)));
    file.read(reinterpret_cast<char*>(system.b.data()), static_cast<std::streamsize>(system.b.size() * sizeof(double)));
    if (!file) throw std::runtime_error("truncated system file " + path);
    return system;
}
