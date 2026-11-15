#include "processor.hpp"

#include <opencv2/imgcodecs.hpp>

cv::Mat load_image(const std::string& path, bool gray) {
    cv::Mat img = cv::imread(path, gray ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR);
    if (img.empty()) throw std::runtime_error("Cannot read image: " + path);
    return img;
}
