#pragma once

#include <string>
#include <type_traits>
#include <stdexcept>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// Reads an image from disk; throws std::runtime_error if it cannot be read.
cv::Mat load_image(const std::string& path, bool gray = true);

// Accepts a file path or a cv::Mat. Returns a single-channel CV_8U image.
template<typename T>
cv::Mat prepare_image(const T& image) {
    cv::Mat img;
    if constexpr (std::is_convertible_v<T, std::string>) {
        img = load_image(image, true);
    } else {
        img = image;
        if (img.empty()) throw std::runtime_error("Empty image");
        if (img.channels() == 3) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        else if (img.channels() == 4) cv::cvtColor(img, img, cv::COLOR_BGRA2GRAY);
        if (img.depth() != CV_8U) img.convertTo(img, CV_8U);
    }
    return img;
}
