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
cv::Size dst_size(60, 60);
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
