#pragma once
#include <opencv2/opencv.hpp>

enum class raw_format { RGGB, BGGR, GRBG, GBRG, BAGR };

class debayer
{
public:
    debayer(raw_format format);
    void raw2bgr(cv::Mat &src, cv::Mat &dst);
    void raw2bgra(cv::Mat &src, cv::Mat &dst);
    void bgr2raw(const cv::Mat &bgr, cv::Mat &raw, raw_format format);

private:
    raw_format format;
};

