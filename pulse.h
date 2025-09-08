#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include "utils.h"


struct pulse_info
{
    std::vector<float> ppg_signal;
    float    ppg_qual = 0;
    int     heartrate = 0;
    std::vector<cv::Rect> bbox_select;
    float    live_status = 0;

};

struct pulse_param
{
//    std::vector<int> bbox_size = std::vector<int>{ 10, 20, 40, 80, 120 };
//    std::vector<float> bbox_step = std::vector<float>{ 1, 1, 1, 0.5, 0.5 };
//    std::vector<int> bbox_size = std::vector<int>{  50, 70, 110 };
//    std::vector<float> bbox_step = std::vector<float>{ 0.5,0.25,0.25 };
//    std::vector<int> bbox_size = std::vector < int > {10, 20, 48};
//    std::vector<float> bbox_step = std::vector < float > {1, 0.5, 0.125};
    std::vector<int> bbox_size = std::vector<int>{ 20, 30};
    std::vector<float> bbox_step = std::vector<float>{0.5, 0.5 };
//    std::vector<int> bbox_size = std::vector<int>{ 10, 20,30};
//    std::vector<float> bbox_step = std::vector<float>{1, 0.5,0.5 };



    float data[3] = { 0.5, 0.8, 0.3 };
    //float data[3] = { 1,1,1 };
    cv::Mat pbv = cv::Mat(1, 3, CV_32FC1, data);

    //float bandpassfilt[2] = { 3, 10 }; //{ 3, 8 };
    //float bandpassfilt[2] = { 10, 40 }; //win=10s
    //float bandpassfilt[2] = { 5, 20 }; //win=10s

//    float bandpassfilt[2] = { 2, 13 }; //40 - 280
    float bandpassfilt[2] = { 3, 12 }; //60 - 240  // i20
    //float bandpassfilt[2] = { 0.1, 5};   // i20
    float overlap_ratio_min = 0.8;
    float size_ratio_max = 2;
    float size_ratio_min = 0.1;

    int select_topK = 20;
    int     fps;
    cv::Size frame_size;
    int     win_len;
    int      live_len;
    int     rate_len;  // 用于计算心率的信号长度
    int heartrate_buffer_len;
    float snr_threshold = 0.5;
    float ratio_threshold = 0.1; // 新增的比例阈值
    int max_heartrate = 180;
    int min_heartrate = 40;
    int max_heartrate_diff = 10;
};
class pulse
{
public:
    //pulse(const cv::Size frame_size, const int fps);
    pulse(const cv::Size frame_size, const int fps);
    pulse_info  detect(const cv::Mat rgb, const cv::Rect faceroi, const bool night_flag);
    std::vector<cv::Rect> bbox, bbox_select, bbox_topK_bgr, bbox_topK_g;

    pulse_param param;
    cv::Mat    PPG_bgr;
    float      smoothed_ratio;
private:
    std::vector<float> ratio_trace;
    void       gen_bbox(const int frame_width, const int frame_height, std::vector<cv::Rect>& bbox);
    cv::Mat    gen_color_triplet(const cv::Mat rgb, const std::vector<cv::Rect> bbox);
    void       concatenate(const cv::Mat val, cv::Mat& trace, int len);
    cv::Vec3f   get_bbox_sum(const cv::Mat mat, int x1, int y1, int x2, int y2);

    void       calc_heartrate(const cv::Mat pulse, int& heartrate);

    cv::Mat   extract_ppg(cv::Mat& rgb_trace, const bool night_flag);
    cv::Mat    gen_snr(const cv::Mat trace);
    cv::Mat gen_spectrum(const cv::Mat trace);

    void       overlap_add(const std::vector<float> val, std::vector<float>& buf, int len);
    void       overlap_add(const cv::Mat val, cv::Mat& buf, int len);

    cv::Mat create_hann(int size);

    void    bandpass(cv::Mat& trace, const float band[2]);
    void    update_rate_trace(const cv::Mat val, cv::Mat& trace, int len);

    cv::Mat hann, hann_mat;
    cv::Mat rgb_trace;
    cv::Mat heartrate_trace;
    cv::Mat temp_buf;
};
