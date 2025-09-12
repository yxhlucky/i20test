#pragma once

#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include "utils.h"

constexpr float INV_255 = 1.0f / 255.0f;

struct breath_info {
    std::vector<float> breath_signal;
    std::vector<float> activity_signal;
    std::vector<float> breath_rates;
    std::vector <cv::Rect> bbox_select;
    // 默认构造函数
    breath_info() {
        // 初始化每个属性为空或默认值
        breath_signal = std::vector<float>();
        activity_signal = std::vector<float>();
        breath_rates = std::vector<float>();
        bbox_select = std::vector<cv::Rect>();  // 初始化为一个空的矩形列表
    }
};

struct breath_param {
    int fps;
    cv::Size frame_size;

//    std::vector<int> bbox_size = std::vector<int>{  8, 12,32 };
//    std::vector<int> bbox_size = std::vector<int>{  8, 12,32 };
//    std::vector<float> bbox_step = std::vector<float>{  1, 0.5,0.5 };
    std::vector<int> bbox_size = std::vector<int>{8};
    std::vector<float> bbox_step = std::vector<float>{1};

    float data[3] = {0.5, 0.0, -0.5};
    cv::Mat kernel_dy = cv::Mat(3, 1, CV_32FC1, data);
    cv::Mat kernel_dx = cv::Mat(1, 3, CV_32FC1, data);


    int motion_len;
    int localtrace_resp_len;
    int localtrace_acti_len;

    cv::Mat detrend_mat;
    cv::Mat hann;

    int globaltrace_resprate_len;
    int globaltrace_activity_len;

    float activity_thr_low; // activity threshold low
    float activity_thr_hig; // activity threshold high
    float qual_thr; // 5;
    float chestroi_thr;
    float chest_present_thr; // 2 min
    float rate_activity_thr = 3;

    int max_breathrate = 99;
    int min_breathrate = 5;
    int max_activity = 10;
    int breathrate_minmax_thr = 3;

    int breath_signal_len;
    int activity_signal_len;
    int breathrate_len;

    std::vector <cv::Rect> bbox;

};


class breath {
public:
    breath(const cv::Size frame_size, const int fps);

    breath_info detect(const cv::Mat &rgb);

    cv::Mat plot_heatmap(const cv::Mat &frame, cv::Mat &softmask, const cv::Rect &roi);

    cv::Mat plot_signal(const cv::Mat &im, const std::vector<float> &breath_signal);

    float motion;

    void reset();

    std::vector <cv::Rect> bbox_select;
private:
    void gen_bbox(const int &frame_width, const int &frame_height, std::vector <cv::Rect> &bbox);

    void update_frame_buffer(const cv::Mat &frame, std::deque <cv::Mat> &imbuf);

    std::vector <cv::Mat> gen_y_shifts(const cv::Mat &frame_pre, const cv::Mat &frame_cur, const cv::Rect& roi);

    std::vector <cv::Mat> gen_xy_shifts(const std::deque <cv::Mat> &imbuf);

    cv::Mat gen_breath_trace(const cv::Mat& dx, const cv::Mat& dy);

    //cv::Mat gen_breath_trace(const cv::Mat localshift_ang, const cv::Rect faceroi, const cv::Rect chestroi);
    float gen_activity_trace(const cv::Mat &localdt);

    void gen_local_traces(const cv::Mat& localshift_ang, cv::Mat &localtrace_resp, cv::Mat &localtrace_acti);

    void gen_global_breath(const cv::Mat &trace, cv::Mat &breath);

    void detrend(cv::Mat &trace, const cv::Mat& A);

    cv::Mat createA(const int &N, const float &lambda);

    cv::Mat createC(const int &framelen, const float &lambda_low, const float &lambda_hig);

    cv::Mat refine_mask(const cv::Mat& mask);

    cv::Mat gen_spectrum(const cv::Mat &trace);

    cv::Mat create_hann(int &size);

    int calc_breathrate(const cv::Mat &breath, const float &activity);

    void calc_activity(const cv::Mat& traces, float &activity);

    void update_buf(const cv::Mat &val, cv::Mat &buf, int &len, bool cumsum_flag, bool overlapadd_flag);

    void modify_breathshape(cv::Mat &trace);

    void concatenate(const cv::Mat &val, std::vector<float> &signal, const int &len);

    void concatenate(const float &val, std::vector<float> &signal, const int &len);

    void gen_accumap();

    cv::Rect update_chestroi(const cv::Mat &frame, const cv::Rect &faceroi);



    /***************************************************************/
    breath_param param;
    cv::Mat breath_trace_ref;
    cv::Mat localtrace_shift_dx, localtrace_shift_dy, localtrace_resp_dx, localtrace_resp_dy, localtrace_resp, localtrace_acti;
    /***************************************************************/
    cv::Mat globaltrace_resprate;
    cv::Mat globaltrace_activity;
    cv::Mat globaltrace_quality;
    /***************************************************************/

    /***************************************************************/
    cv::Mat accumap_global;
    /***************************************************************/
    cv::Mat softmask;
    std::deque <cv::Mat> imbuf;

    std::vector<float> breath_signal;
    std::vector<float> breath_rates;
    std::vector<float> activity_signal;


//	int chest_flag;
};