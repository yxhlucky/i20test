//
// Created by yangxh on 25-6-18.
//

#ifndef ALGORITHM_TEST_RUN_H
#define ALGORITHM_TEST_RUN_H


#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include "pulse.h"
#include "select_roi.h"
#include "read_file.h"
#include "bin2mat.h"
#include "debayer.h"
#include "breath.h"
#include "utils.h"

class YuNet
{
public:
    YuNet(const std::string& model_path,
          const cv::Size& input_size = cv::Size(320, 320),
          float conf_threshold = 0.3f,
          float nms_threshold = 0.2f,
          int top_k = 10000,
          int backend_id = cv::dnn::DNN_BACKEND_OPENCV,
          int target_id = cv::dnn::DNN_TARGET_CPU)
            : model_path_(model_path), input_size_(input_size),
              conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
              top_k_(top_k), backend_id_(backend_id), target_id_(target_id)
    {
        model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
    }

    void setInputSize(const cv::Size& input_size)
    {
        input_size_ = input_size;
        model->setInputSize(input_size_);
    }

    cv::Mat infer(const cv::Mat& image)
    {
        cv::Mat res;
        model->detect(image, res);
        return res;
    }

private:
    cv::Ptr<cv::FaceDetectorYN> model;
    std::string model_path_;
    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    int top_k_;
    int backend_id_;
    int target_id_;
};

cv::Mat visualize(const cv::Mat& image, const cv::Mat& faces)
{
    static cv::Scalar box_color{0, 255, 0};
    auto output_image = image.clone();

    for (int i = 0; i < faces.rows; ++i)
    {
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        cv::rectangle(output_image, cv::Rect(x1, y1, w, h), box_color, 2);
    }
    return output_image;
}

std::vector<cv::Rect> getFaceBoxes(const cv::Mat& faces)
{
    std::vector<cv::Rect> faceBoxes;

    for (int i = 0; i < faces.rows; ++i)
    {
        int x1 = static_cast<int>(faces.at<float>(i, 0));
        int y1 = static_cast<int>(faces.at<float>(i, 1));
        int w = static_cast<int>(faces.at<float>(i, 2));
        int h = static_cast<int>(faces.at<float>(i, 3));
        faceBoxes.emplace_back(x1, y1, w, h);
    }

    return faceBoxes;
}
//#include "HRV.h"
#define WIN_LEN_HRV 300
vector<double> bandwidth = {0.7, 3.0};
int frameCount_HRV = 0;
//HRV HRV_processor(bandwidth, FPS, 60);
std::vector<std::string> files;
std::vector<cv::Rect> rois;
std::vector<cv::Rect> rois_chest;
debayer db(raw_format::RGGB);
RectangleDrawer rd;
int FPS = 30;
int signal_len = FPS * 10;
cv::Size dst_size(80, 80);
int bpm_value = 0;
int ppg_bpm_value = 0;
class DataProcessor {
public:
    DataProcessor(size_t buffer_length) : buffer_length(buffer_length) {}

    void addDataPoint(int data_point) {
        data_buffer.push_back(data_point);
        if (data_buffer.size() > buffer_length) {
            data_buffer.pop_front();
        }
    }

    double calculateAverage() const {
        // 计算非零元素的总和
        double sum = std::accumulate(data_buffer.begin(), data_buffer.end(), 0.0,
                                     [](double a, int b) { return b != 0 ? a + b : a; });

        // 计算非零元素的数量
        int non_zero_count = std::count_if(data_buffer.begin(), data_buffer.end(), [](int i) { return i != 0; });

        // 计算非零元素的平均值，避免除以零
        return non_zero_count > 0 ? sum / non_zero_count : 0;
    }

private:
    std::deque<int> data_buffer;
    size_t buffer_length;
};


#endif //ALGORITHM_TEST_RUN_H
