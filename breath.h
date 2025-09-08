#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <fstream>
struct breath_info
{
    std::vector<float> breath_signal;
    std::vector<float> activity_signal;
    std::vector<float> breath_rates;
    cv::Rect faceroi;
    cv::Rect chestroi;
    std::vector<cv::Rect> bbox_select;
    // 1) 提供一个静态空实例（懒汉式）
    static breath_info empty() {
        return breath_info{};
    }
};

struct breath_param
{
    int fps;
    cv::Size frame_size;

//  std::vector<int> bbox_size = std::vector<int>{ 10, 20, 40, 80 };
//  std::vector<float> bbox_step = std::vector<float>{ 1, 1, 1, 0.5 };
//    std::vector<int> bbox_size = std::vector<int>{ 20, 80, 100 };
//    std::vector<float> bbox_step = std::vector<float>{ 1, 0.5,0.25 };
    std::vector<int> bbox_size = std::vector<int>{20};
    std::vector<float> bbox_step = std::vector<float>{ 1};
//    std::vector<int> bbox_size = std::vector<int>{8};
//    std::vector<float> bbox_step = std::vector<float>{ 1};

    float data[3] = { 0.5, 0.0, -0.5 };
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
    int breathrate_minmax_thr; // 5 bpm contrast
    float qual_thr; // 5;
    float chestroi_thr;
    float chest_present_thr; // 2 min

    std::vector<cv::Rect> bbox;
};




class breath
{
public:
    breath(const cv::Size frame_size, const int fps);
    breath_info detect(const cv::Mat& rgb, const cv::Rect chestroi);
    cv::Mat plot_heatmap(const cv::Mat frame, cv::Mat softmask, const cv::Rect roi);
    cv::Mat plot_signal(const cv::Mat im, const std::vector<float> breath_signal);
    float    motion;
    std::vector<cv::Rect> bbox_select;
private:

    std::vector<int> bbox_region; // 存储每个bbox所属的区域(0=上, 1=左, 2=下, 3=右)
    std::vector<float> region_votes; // 四个区域的累积投票
    int best_region; // 当前最佳区域
    int vote_reset_counter;
    int vote_reset_interval; // 重置间隔（帧数）
    void overlap_add(const cv::Mat val, cv::Mat &buf, int len);

    void    gen_bbox(const int frame_width, const int frame_height, std::vector<cv::Rect>& bbox);
    void    update_frame_buffer(cv::Mat frame, std::deque<cv::Mat>& imbuf);
    std::vector<cv::Mat> gen_y_shifts(const cv::Mat frame_pre, const cv::Mat frame_cur, const cv::Rect roi);
    std::vector<cv::Mat> gen_xy_shifts(const std::deque<cv::Mat> imbuf, const cv::Rect roi);

    cv::Mat gen_breath_trace(const cv::Mat dx, const cv::Mat dy, const cv::Rect chestroi);
    //cv::Mat gen_breath_trace(const cv::Mat localshift_ang, const cv::Rect faceroi, const cv::Rect chestroi);
    float gen_activity_trace(const cv::Mat localdt);

    void    gen_local_traces(const cv::Mat localshift_ang, cv::Mat& localtrace_resp, cv::Mat& localtrace_acti);
    void    gen_global_breath(const cv::Mat trace, cv::Mat& breath);

    void    detrend(cv::Mat& trace, const cv::Mat A);
    cv::Mat createA(const int N, const float lambda);
    cv::Mat createC(const int framelen, const float lambda_low, const float lambda_hig);
    cv::Mat refine_mask(const cv::Mat mask);

    cv::Mat gen_spectrum(const cv::Mat trace);
    cv::Mat create_hann(int size);
    int calc_breathrate(const cv::Mat breath, const float activity);
    void    calc_activity(const cv::Mat traces, float& activity);
    void    update_buf(const cv::Mat val, cv::Mat& buf, int len, bool cumsum_flag, bool overlapadd_flag);
    void    modify_breathshape(cv::Mat& trace);

    void concatenate(const cv::Mat val, std::vector<float>& signal, const int len);
    void concatenate(const float val, std::vector<float>& signal, const int len);
    void gen_accumap();
    cv::Rect update_chestroi(const cv::Mat frame, const cv::Rect faceroi);

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
    std::deque<cv::Mat> imbuf;

    std::vector<float> breath_signal;
    std::vector<float> breath_rates;
    std::vector<float> activity_signal;
    cv::Mat temp_buf;

    static constexpr int SLIDE_LEN = 225;
    float slide_sum[4]   = {0};          // 当前窗口内总和
    float slide_buf[4][SLIDE_LEN] = {0}; // 环形缓冲
    int   slide_idx = 0;                 // 写指针

//  int chest_flag;
};
