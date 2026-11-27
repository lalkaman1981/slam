#include "rgbd_pipeline.hpp"

#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
struct ImageEntry {
    double time;
    std::filesystem::path path;
};
struct ImagePair {
    ImageEntry rgb, depth;
};

std::vector<ImageEntry> readImageList(const std::filesystem::path &path) {
    std::ifstream file(path);
    if (!file) throw std::runtime_error("Cannot open " + path.string());
    std::vector<ImageEntry> entries;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos ||
            line[line.find_first_not_of(" \t\r")] == '#')
            continue;
        std::istringstream row(line);
        double time;
        std::string name;
        if (!(row >> time >> name) || !std::isfinite(time) ||
            (!entries.empty() && time <= entries.back().time))
            throw std::runtime_error("Invalid/non-increasing timestamp list: " + path.string());
        entries.push_back({time, path.parent_path() / name});
    }
    return entries;
}

std::vector<ImagePair> associate(const std::filesystem::path &dataset) {
    const auto rgb = readImageList(dataset / "rgb.txt");
    const auto depth = readImageList(dataset / "depth.txt");
    std::vector<ImagePair> pairs;
    size_t j = 0;
    for (const auto &image : rgb) {
        while (j + 1 < depth.size() &&
               std::abs(depth[j + 1].time - image.time) < std::abs(depth[j].time - image.time))
            ++j;
        if (j < depth.size() && std::abs(depth[j].time - image.time) <= 0.02)
            pairs.push_back({image, depth[j++]});
    }
    return pairs;
}

void usage() {
    std::cout << "Usage: slam_rgbd DATASET OUTPUT [--mode serial|serial_simd|parallel]\n"
                 "       [--fx N --fy N --cx N --cy N] [--features N] [--max-frames N]\n"
                 "       [--min-initial-points N] [--depth-scale N]\n\n"
                 "DATASET contains TUM rgb.txt/depth.txt plus registered images.\n"
                 "Defaults: fx=525 fy=525 cx=319.5 cy=239.5, depth-scale=5000 units/m.\n"
                 "Pass the sequence calibration; image size comes from the first RGB frame.\n"
                 "The mode selects mapping only; imported extraction and tracking are serial.\n";
}
} // namespace

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            usage();
            return 0;
        }
        if (argc < 3) {
            usage();
            return 1;
        }
        slam::PipelineConfig config;
        size_t maxFrames = 0;
        double depthScale = 5000.0;
        for (int i = 3; i < argc; ++i) {
            const std::string option = argv[i];
            if (++i >= argc) throw std::invalid_argument("Missing value for " + option);
            const std::string value = argv[i];
            if (option == "--mode") {
                if (value == "serial")
                    config.mode = slam::ExecutionMode::Serial;
                else if (value == "serial_simd")
                    config.mode = slam::ExecutionMode::SerialSimd;
                else if (value == "parallel")
                    config.mode = slam::ExecutionMode::Parallel;
                else
                    throw std::invalid_argument("Unknown mode: " + value);
            } else if (option == "--fx")
                config.camera.fx = std::stod(value);
            else if (option == "--fy")
                config.camera.fy = std::stod(value);
            else if (option == "--cx")
                config.camera.cx = std::stod(value);
            else if (option == "--cy")
                config.camera.cy = std::stod(value);
            else if (option == "--features")
                config.maxFeatures = std::stoi(value);
            else if (option == "--min-initial-points")
                config.mapping.minInitialPoints = std::stoi(value);
            else if (option == "--max-frames") {
                const int n = std::stoi(value);
                if (n < 0) throw std::invalid_argument("Negative frame limit");
                maxFrames = static_cast<size_t>(n);
            } else if (option == "--depth-scale")
                depthScale = std::stod(value);
            else
                throw std::invalid_argument("Unknown option: " + option);
        }
        if (!std::isfinite(depthScale) || depthScale <= 0) throw std::invalid_argument("Invalid depth scale");
        auto pairs = associate(argv[1]);
        if (pairs.empty()) throw std::runtime_error("No RGB/depth pairs within 20 ms");
        if (maxFrames > 0 && maxFrames < pairs.size()) pairs.resize(maxFrames);
        const auto first = cv::imread(pairs.front().rgb.path.string(), cv::IMREAD_GRAYSCALE);
        if (first.empty()) throw std::runtime_error("Cannot load first RGB image");
        config.camera.width = first.cols;
        config.camera.height = first.rows;
        slam::RgbdPipeline pipeline(config);
        const std::filesystem::path output(argv[2]);
        std::filesystem::create_directories(output);
        std::ofstream stats(output / "frames.csv");
        if (!stats) throw std::runtime_error("Cannot write frame statistics");
        stats
            << "timestamp,status,features,matches,inliers,keyframe,loop,features_ms,tracking_ms,mapping_ms\n";
        size_t tracked = 0, loops = 0;
        double featureSeconds = 0, trackingSeconds = 0, mappingSeconds = 0;
        for (const auto &pair : pairs) {
            const auto image = cv::imread(pair.rgb.path.string(), cv::IMREAD_GRAYSCALE);
            auto depth = cv::imread(pair.depth.path.string(), cv::IMREAD_UNCHANGED);
            if (depth.empty() || depth.type() != CV_16UC1)
                throw std::runtime_error("Expected 16-bit depth: " + pair.depth.path.string());
            if (depthScale != DEPTH_SCALE) {
                // Preserve invalid/out-of-range depth as zero rather than saturating it.
                for (int y = 0; y < depth.rows; ++y)
                    for (int x = 0; x < depth.cols; ++x) {
                        auto &raw = depth.at<uint16_t>(y, x);
                        const double metres = raw / depthScale;
                        raw = metres >= MIN_DEPTH && metres <= MAX_DEPTH
                                  ? static_cast<uint16_t>(std::lround(metres * DEPTH_SCALE))
                                  : 0;
                    }
            }
            const auto result = pipeline.process(image, depth, pair.rgb.time);
            tracked += result.tracked;
            loops += result.mapping.loop.closed;
            featureSeconds += result.featureSeconds;
            trackingSeconds += result.trackingSeconds;
            mappingSeconds += result.mappingSeconds;
            const char *status = result.initialized   ? "INITIALIZED"
                                 : result.relocalized ? "RELOCALIZED"
                                 : result.tracked     ? "OK"
                                                      : "LOST";
            stats << std::fixed << std::setprecision(6) << pair.rgb.time << ',' << status << ','
                  << result.features << ',' << result.matches << ',' << result.inliers << ','
                  << result.mapping.keyframeId << ',' << result.mapping.loop.closed << ','
                  << result.featureSeconds * 1000 << ',' << result.trackingSeconds * 1000 << ','
                  << result.mappingSeconds * 1000 << '\n';
        }
        stats.close();
        if (!stats || !slam::exportMap(pipeline.map(), (output / "map.ply").string()) ||
            !slam::exportTrajectory(pipeline.map(), (output / "trajectory.txt").string()) ||
            !slam::exportKeyframeTrajectory(pipeline.map(), (output / "keyframe_trajectory.txt").string()))
            throw std::runtime_error("Failed to write output files");
        std::cout << "Tracked " << tracked << '/' << pairs.size()
                  << " frames, keyframes=" << pipeline.map().keyframes.size()
                  << ", landmarks=" << pipeline.map().landmarks.size() << ", loops=" << loops << '\n'
                  << "Seconds: features=" << featureSeconds << " tracking=" << trackingSeconds
                  << " mapping=" << mappingSeconds << '\n'
                  << "Output: " << output << '\n';
        if (tracked < 2) {
            std::cerr << "Insufficient tracking: check calibration, depth scale and texture.\n";
            return 2;
        }
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "slam_rgbd: " << error.what() << '\n';
        return 1;
    }
}
