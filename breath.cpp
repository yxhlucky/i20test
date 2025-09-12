#include "breath.h"

breath::breath(const cv::Size frame_size, const int fps) {
    param.fps = fps;
    param.frame_size = frame_size;

    gen_bbox(param.frame_size.width, param.frame_size.height, param.bbox);

    param.motion_len = 3 * fps; // 3 sec length
    param.localtrace_resp_len = 15 * fps; // 15 sec length
    param.localtrace_acti_len = 10 * fps; // 10 sec length

    param.detrend_mat = createC(param.motion_len, 0.1, 1000);
    param.hann = create_hann(param.localtrace_resp_len);

    param.globaltrace_resprate_len = fps * 5; // 5-sec buffer
    param.globaltrace_activity_len = fps * 1; // 3-sec buffer

    param.activity_thr_low = 0.1; // activity threshold low
    param.activity_thr_hig = 5; // activity threshold high
    param.qual_thr = 5; // 5;
    param.chestroi_thr = 0.8;
    param.chest_present_thr = fps * 60 * 2; // 2 min

    param.breath_signal_len = 1 * 60 * fps; // 1 min buffer
    param.activity_signal_len = 1 * 60 * fps; // 1 min buffer
    param.breathrate_len = 1 * 60 * fps; // 1 min buffer


}

breath_info breath::detect(const cv::Mat &rgb) {
    // update frame buffer
    update_frame_buffer(rgb, imbuf);

    // generate local shifts
    std::vector <cv::Mat> shifts = gen_xy_shifts(imbuf);

    // generate breath trace
    cv::Mat breath_trace = gen_breath_trace(shifts.at(0), shifts.at(1));

    // generate activity
    float activity = gen_activity_trace(shifts.at(2));

    // generate breath rate
    int breathrate = calc_breathrate(breath_trace, activity);

    modify_breathshape(breath_trace);

    // set breath_info
    breath_info bi;
    // temporal accumulate
    concatenate(breath_trace, bi.breath_signal, param.breath_signal_len); // generate breath signal
    concatenate(activity, bi.activity_signal, param.activity_signal_len); // generat activity signal
    concatenate((float) breathrate, bi.breath_rates, param.breathrate_len); // generat activity signal
    bi.bbox_select = bbox_select;
    return bi;
}

void breath::update_frame_buffer(const cv::Mat &frame, std::deque <cv::Mat> &imbuf) {
    if (frame.empty()) return;

    cv::Mat gray, grayf;

    // check frame channel
    if (frame.channels() > 1) {
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        gray.convertTo(grayf, CV_32FC1, INV_255); // convert uint8 to float32, 255 to 1
    } else {
        frame.convertTo(grayf, CV_32FC1, INV_255); // convert uint8 to float32, 255 to 1
    }

    imbuf.push_back(grayf);
    if (imbuf.size() == 4) // N-1 frames buffer
    {
        imbuf.pop_front();
    }
}

std::vector <cv::Mat> breath::gen_xy_shifts(const std::deque <cv::Mat> &imbuf) {
    if (imbuf.size() < 2) {

        static cv::Mat localdx = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        static cv::Mat localdy = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        static cv::Mat localdt = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);

        return std::vector < cv::Mat > {localdx, localdy, localdt};
    } else {
        cv::Mat frame_sub = imbuf.at(0) - imbuf.at(imbuf.size() - 1);
        cv::Mat frame_add = (imbuf.at(0) + imbuf.at(imbuf.size() - 1)) / 2;

        cv::Mat dx, dy, dt;
        cv::filter2D(frame_add, dx, CV_32FC1, param.kernel_dx, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(frame_add, dy, CV_32FC1, param.kernel_dy, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);

        //mxu::divide(mxu_In_Mat_frame_sub,mxu_In_Mat_frame_add + 1e-3,mxu_Mat_frame_dt);
        cv::divide(frame_sub, frame_add + 1e-3, dt);

        cv::Mat dydy, dydt, dxdx, dxdt, dxdy;
        cv::Mat idydy, idydt, idxdx, idxdt, idxdy, idt;
        cv::multiply(dy, dy, dydy);
        cv::integral(dydy, idydy, CV_32FC1);
        cv::multiply(dy, dt, dydt);
        cv::integral(dydt, idydt, CV_32FC1);
        cv::multiply(dx, dx, dxdx);
        cv::integral(dxdx, idxdx, CV_32FC1);
        cv::multiply(dx, dt, dxdt);
        cv::integral(dxdt, idxdt, CV_32FC1);
        cv::multiply(dx, dy, dxdy);
        cv::integral(dxdy, idxdy, CV_32FC1);
        cv::integral(dt, idt, CV_32FC1);

        //cv::Mat localda = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        static cv::Mat localdx = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        static cv::Mat localdy = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        static cv::Mat localdt = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);

        for (int i = 0; i < param.bbox.size(); i++) {
            int x1 = param.bbox.at(i).x;
            int x2 = param.bbox.at(i).x + param.bbox.at(i).width - 1;
            int y1 = param.bbox.at(i).y;
            int y2 = param.bbox.at(i).y + param.bbox.at(i).height - 1;

            float sum_dydt = idydt.at<float>(y2 + 1, x2 + 1) + idydt.at<float>(y1, x1) - idydt.at<float>(y2 + 1, x1) -
                             idydt.at<float>(y1, x2 + 1);
            float sum_dydy = idydy.at<float>(y2 + 1, x2 + 1) + idydy.at<float>(y1, x1) - idydy.at<float>(y2 + 1, x1) -
                             idydy.at<float>(y1, x2 + 1);
            float sum_dxdt = idxdt.at<float>(y2 + 1, x2 + 1) + idxdt.at<float>(y1, x1) - idxdt.at<float>(y2 + 1, x1) -
                             idxdt.at<float>(y1, x2 + 1);
            float sum_dxdx = idxdx.at<float>(y2 + 1, x2 + 1) + idxdx.at<float>(y1, x1) - idxdx.at<float>(y2 + 1, x1) -
                             idxdx.at<float>(y1, x2 + 1);
            float sum_dxdy = idxdy.at<float>(y2 + 1, x2 + 1) + idxdy.at<float>(y1, x1) - idxdy.at<float>(y2 + 1, x1) -
                             idxdy.at<float>(y1, x2 + 1);
            float sum_dt = idt.at<float>(y2 + 1, x2 + 1) + idt.at<float>(y1, x1) - idt.at<float>(y2 + 1, x1) -
                           idt.at<float>(y1, x2 + 1);

            float ux =
                    (-sum_dxdy * sum_dydt + sum_dydy * sum_dxdt) / (sum_dxdx * sum_dydy - sum_dxdy * sum_dxdy + 1e-6);
            float uy =
                    (-sum_dxdy * sum_dxdt + sum_dxdx * sum_dydt) / (sum_dxdx * sum_dydy - sum_dxdy * sum_dxdy + 1e-6);

//            std::complex<float> complex(ux, uy);
            //localda.at<float>(i, 0) = std::arg(complex)* std::abs(complex);
            //localda.at<float>(i, 0) = uy;// std::arg(complex)* std::abs(complex);
            localdx.at<float>(i, 0) = ux;
            localdy.at<float>(i, 0) = uy;
            localdt.at<float>(i, 0) = 100 * sum_dt / param.bbox.at(i).area();
        }
        //localshift_mag = localshift_mag / (1e-3 + cv::norm(localshift_mag, cv::NORM_L2));
        //cv::multiply(localshift_ang, localshift_mag, localshift_ang);

        return std::vector < cv::Mat > {localdx, localdy, localdt};
    }
}

//cv::Mat combineXY(const cv::Mat &trace_dx, const cv::Mat& trace_dy) {
//    cv::Mat A;
//    cv::vconcat(trace_dx, trace_dy, A);
//
//    cv::Mat R;
//
//    mxu::Mat mxu_In_Mat_A(A);
//    mxu::_InputArray mxu_In_Arr_A(mxu_In_Mat_A);
//
//    mxu::Mat mxu_Out_Mat_R(R);
//    mxu::_OutputArray mxu_Out_Arr_R(mxu_Out_Mat_R);
//    mxu::gemm(mxu_In_Arr_A, mxu_In_Arr_A, 1, mxu::Mat(), 0, mxu_Out_Arr_R, mxu::GEMM_2_T);
//    mxu_Out_Mat_R = mxu_Out_Arr_R.getMat();
//    R = mxu_Out_Mat_R.toCvMat();
//
//    //cv::gemm(A, A, 1, cv::Mat(), 0, R, cv::GEMM_2_T);
//
//    cv::Mat w, u, vt;
//    cv::SVDecomp(R, w, u, vt, cv::SVD::FULL_UV); //cv::DECOMP_QR
//
//    cv::Mat S = u.col(0) * u.col(0).t() * A;
//    cv::reduce(S, S, 0, cv::REDUCE_AVG);
//
//    return S;
//}
//
//cv::Mat combineXY2(const cv::Mat& trace_dx, const cv::Mat &trace_dy) {
//    static cv::Scalar dx_avg, dx_std, dy_avg, dy_std;
//    cv::meanStdDev(trace_dx, dx_avg, dx_std);
//    cv::meanStdDev(trace_dy, dy_avg, dy_std);  // 4-6ms
//
//    static cv::Mat trace_dx_n, trace_dy_n;
//    cv::subtract(trace_dx, (float) dx_avg[0], trace_dx_n);
//    cv::subtract(trace_dy, (float) dy_avg[0], trace_dy_n);  // 6-9ms
//
//    float sign = trace_dx_n.dot(trace_dy_n) > 0 ? 1 : -1;  // 约1ms
//    float norm = sqrt(dx_std[0] * dx_std[0] + dy_std[0] * dy_std[0]);
//    float xn = dx_std[0] / norm;
//    float yn = (dy_std[0] * sign) / norm;
//    float a = xn * xn + xn * yn;
//    float b = yn * yn + xn * yn;
//
//    cv::Mat S = a * trace_dx_n + b * trace_dy_n;  // 6-10 ms
//
//    return S;
//}

cv::Mat combineXY3(const cv::Mat& trace_dx, const cv::Mat& trace_dy) {
    static cv::Mat trace_dx_avg, trace_dy_avg;
    cv::reduce(trace_dx, trace_dx_avg, 1, cv::REDUCE_AVG);
    cv::reduce(trace_dy, trace_dy_avg, 1, cv::REDUCE_AVG);

    static cv::Mat trace_dx_avg_mat, trace_dy_avg_mat;
    cv::repeat(trace_dx_avg, 1, trace_dx.cols, trace_dx_avg_mat);
    cv::repeat(trace_dy_avg, 1, trace_dy.cols, trace_dy_avg_mat);

    static cv::Mat trace_dx_n, trace_dy_n;
    cv::subtract(trace_dx, trace_dx_avg_mat, trace_dx_n);
    cv::subtract(trace_dy, trace_dy_avg_mat, trace_dy_n);

    static cv::Mat trace_dx_pow, trace_dy_pow;
    cv::pow(trace_dx_n, 2, trace_dx_pow);
    cv::pow(trace_dy_n, 2, trace_dy_pow);

    static cv::Mat trace_dx_std, trace_dy_std;
    cv::reduce(trace_dx_pow, trace_dx_std, 1, cv::REDUCE_AVG);
    cv::sqrt(trace_dx_std, trace_dx_std);
    cv::reduce(trace_dy_pow, trace_dy_std, 1, cv::REDUCE_AVG);
    cv::sqrt(trace_dy_std, trace_dy_std);

    cv::Mat norm_sq;
    cv::add(trace_dx_std.mul(trace_dx_std), trace_dy_std.mul(trace_dy_std), norm_sq); //避免 pow 和 sqrt
    cv::Mat norm;
    cv::sqrt(norm_sq, norm);
    norm += 1e-6;
    cv::Mat Wx = trace_dx_std / norm; // 广播
    cv::Mat Wy = trace_dy_std / norm; // 广播

    static cv::Mat trace_dxdy_prod;
    cv::multiply(trace_dx_n, trace_dy_n, trace_dxdy_prod);

    static cv::Mat trace_dxdy_prod_sum;
    cv::reduce(trace_dxdy_prod, trace_dxdy_prod_sum, 1, cv::REDUCE_SUM);

    static cv::Mat mask;
    cv::compare(trace_dxdy_prod_sum, 0, mask, cv::CMP_LT);

    cv::Mat sign = cv::Mat::ones(mask.rows, 1, CV_32FC1);
    sign.setTo(-1, mask);

    cv::multiply(Wy, sign, Wy);

    static cv::Mat Wy2, WxWy, Wx2;
    cv::multiply(Wx, Wx, Wx2);
    cv::multiply(Wy, Wy, Wy2);
    cv::multiply(Wx, Wy, WxWy);

    static cv::Mat a, b;
    cv::add(Wx2, WxWy, a);
    cv::add(Wy2, WxWy, b);

    static cv::Mat a_mat, b_mat;
    cv::repeat(a, 1, trace_dx.cols, a_mat);
    cv::repeat(b, 1, trace_dy.cols, b_mat);

    cv::multiply(a_mat, trace_dx_n, a_mat);
    cv::multiply(b_mat, trace_dy_n, b_mat);

    cv::Mat S;
    cv::add(a_mat, b_mat, S);

    return S;
}

cv::Mat remove_avg_std(const cv::Mat &trace, float bias) {
    static cv::Mat trace_avg;
    cv::reduce(trace, trace_avg, 1, cv::REDUCE_AVG);

    static cv::Mat trace_avg_mat;
    cv::repeat(trace_avg, 1, trace.cols, trace_avg_mat);

    cv::Mat trace_n;
    cv::subtract(trace, trace_avg_mat, trace_n);

    static cv::Mat trace_pow;
    cv::pow(trace_n, 2, trace_pow);

    static cv::Mat trace_std;
    cv::reduce(trace_pow, trace_std, 1, cv::REDUCE_AVG);
    cv::sqrt(trace_std, trace_std);
    cv::add(trace_std, bias, trace_std);

    static cv::Mat trace_std_mat;
    cv::repeat(trace_std, 1, trace.cols, trace_std_mat);

    cv::divide(trace_n, trace_std_mat, trace_n);

    return trace_n;
}


cv::Mat breath::gen_breath_trace(const cv::Mat& dx, const cv::Mat &dy) {
    update_buf(dx, localtrace_shift_dx, param.motion_len, true, false);
    update_buf(dy, localtrace_shift_dy, param.motion_len, true, false);

    // detrend and denoise
    if (localtrace_shift_dx.cols == param.motion_len) {
        // Option 3: STD mat based
        cv::Mat localtrace_shift = combineXY3(localtrace_shift_dx, localtrace_shift_dy);

        // remove non-respiratory components
        detrend(localtrace_shift, param.detrend_mat);

        update_buf(localtrace_shift, localtrace_resp, param.localtrace_resp_len, false, true);
    }

    if (localtrace_resp.cols != param.localtrace_resp_len) {
        return cv::Mat();
    }

    // normalize local signals
    cv::Mat localtrace_sel_n = remove_avg_std(localtrace_resp, 1);

    // correct sign
    cv::Mat trace_ac;
    if (!breath_trace_ref.empty()) {
        cv::gemm(localtrace_sel_n, breath_trace_ref, 1, cv::Mat(), 0, trace_ac, cv::GEMM_2_T);
    } else {
        trace_ac = cv::Mat::ones(localtrace_sel_n.rows, 1, CV_32FC1);
    }

    cv::normalize(trace_ac, softmask, 1, 0, cv::NORM_L2);

    // generate breath trace
    cv::Mat breath_trace;
    cv::gemm(softmask, localtrace_sel_n, 1, cv::Mat(), 0, breath_trace, cv::GEMM_1_T);

    cv::Scalar avg, stddev;
    cv::meanStdDev(breath_trace, avg, stddev);
    breath_trace = (breath_trace - avg[0]) / (1 + stddev[0]);

    // update reference breath
    cv::hconcat(breath_trace.colRange(1, breath_trace.cols), cv::Mat::zeros(1, 1, CV_32FC1), breath_trace_ref);

    return breath_trace;
}

float breath::gen_activity_trace(const cv::Mat &localdt) {
    // update activity traces
    update_buf(localdt, localtrace_acti, param.localtrace_acti_len, true, false);

    // energy per trace
    cv::Mat energy = cv::Mat(localtrace_acti.rows, 1, CV_32FC1);
    for (int i = 0; i < localtrace_acti.rows; i++) {
        cv::Scalar avg, stddev;
        cv::meanStdDev(localtrace_acti.row(i), avg, stddev);
        energy.at<float>(i, 0) = (float) stddev.val[0];
    }

    // sort energy (max to min)
    cv::Mat energy_sort;
    cv::sort(energy, energy_sort, cv::SORT_DESCENDING + cv::SORT_EVERY_COLUMN);

    // topK 10% average energy [0, 10]
    int topK = (int) (0.5 * energy_sort.rows);
    float energy_topK = (float) cv::mean(energy_sort.rowRange(0, topK))[0];
    float activity_tmp = 1 * (cv::exp(std::sqrt(energy_topK)) - 1);

    if (activity_tmp > param.max_activity) {
        activity_tmp = param.max_activity;
    }

    // temporal average
    update_buf(cv::Mat(1, 1, CV_32FC1, (float) activity_tmp), globaltrace_activity, param.globaltrace_activity_len,
               false, false);

    float activity = (float) cv::mean(globaltrace_activity).val[0];
    motion = (float) cv::mean(globaltrace_activity).val[0];
    return activity;
}

void breath::detrend(cv::Mat &trace, const cv::Mat& A) {
    cv::gemm(trace, A, 1, cv::Mat(), 0, trace);
}

int breath::calc_breathrate(const cv::Mat& breath_trace, const float& activity) {
    if (breath_trace.empty()) {
        return 0;
    }

    // zero mean
    cv::Mat breath_trace_n;
    cv::subtract(breath_trace, cv::mean(breath_trace), breath_trace_n);

    // hanning windowing
    cv::Mat breath_trace_hann;
    cv::multiply(breath_trace_n, param.hann, breath_trace_hann);

    // zero padding
    cv::Mat breath_trace_zp;
    cv::hconcat(breath_trace_hann,
                cv::Mat::zeros(breath_trace_hann.rows, param.fps * 60 - breath_trace_hann.cols, CV_32FC1),
                breath_trace_zp);

    // spectrum
    cv::Mat spec = gen_spectrum(breath_trace_zp);

    // breath rate
    int breathrate = -1;
    double maxval = 0;
    cv::minMaxIdx(spec.t(), 0, &maxval, 0, &breathrate);

    // breath quality
//    float quality = (float) maxval;

    int avg_in_max_bin = 0;
    int bins_above_10_count = 0;
    float stdev = 0;
    // check and update buffer
    if (breathrate >= param.min_breathrate && breathrate <= param.max_breathrate && activity > param.activity_thr_low &&
        activity < param.activity_thr_hig) // || quality > qual_thr)
    {
        update_buf(cv::Mat(1, 1, CV_32FC1, (float) breathrate), globaltrace_resprate, param.globaltrace_resprate_len,
                   false, false);

        /********************  BEGIN  直方图峰值检测  ********************/
        int bins = 94;
        cv::Mat hist;
        int histSize[] = {bins};
        float brRange[] = {float(param.min_breathrate), float(param.max_breathrate)};
        const float *ranges[] = {brRange};
        // 计算呼吸值直方图
        cv::calcHist(&globaltrace_resprate, 1, 0, cv::Mat(), hist, 1, histSize, ranges);

        // 找到最大 bin（呼吸值出现最多的区间）
        int maxBinIndex = std::max_element(hist.begin<float>(), hist.end<float>()) - hist.begin<float>();

        // 计算该 bin 对应的心率区间
        int maxBinRangeStart = cvRound(brRange[0] + maxBinIndex * (brRange[1] - brRange[0]) / bins);
        int maxBinRangeEnd = cvRound(brRange[0] + (maxBinIndex + 1) * (brRange[1] - brRange[0]) / bins);

        // 计算该区间的平均值
        std::vector<float> values_in_max_bin;
        for (int i = 0; i < globaltrace_resprate.cols; ++i) {
            float value = globaltrace_resprate.at<float>(0, i);
            if (value >= maxBinRangeStart && value <= maxBinRangeEnd) {
                values_in_max_bin.push_back(value);
            }
        }

        if (!values_in_max_bin.empty()) {
            avg_in_max_bin = std::accumulate(values_in_max_bin.begin(), values_in_max_bin.end(), 0.0f) /
                             values_in_max_bin.size();
        }

        std::unordered_map<int, int> peak_counts;
        // 遍历直方图，统计各 bin 计数，只有当计数大于 10 时才计入峰值检测
        for (int i = 0; i < bins; i++) {
            int bin_value = cvRound(brRange[0] + i * (brRange[1] - brRange[0]) / bins);
            int bin_count = cvRound(hist.at<float>(i));

            if (bin_count > 10) {  // 当前簇的计数
                peak_counts[bin_value] += bin_count;
                bins_above_10_count++;
            }
        }
        stdev = calculateWeightedStd(peak_counts);
    }
    /********************  END  直方图峰值检测  ********************/
    if (globaltrace_resprate.empty() || stdev > param.rate_activity_thr)
        breathrate = 0;
    else
        breathrate = avg_in_max_bin;

    if (activity <= param.activity_thr_low) // no-subject case
    {
        breathrate = 0;
        globaltrace_resprate = cv::Mat();
    }
    return breathrate;
}

cv::Mat breath::gen_spectrum(const cv::Mat& trace) {
    cv::Mat fft, fft_mag;
    cv::Mat tmp[2] = {trace, cv::Mat::zeros(trace.size(), trace.type())};
    cv::merge(tmp, 2, fft);
    cv::dft(fft, fft, cv::DFT_SCALE + cv::DFT_COMPLEX_OUTPUT + cv::DFT_ROWS);

    cv::split(fft.colRange(0, floor(fft.cols / 2) - 1), tmp);
    cv::magnitude(tmp[0], tmp[1], fft_mag);

    return fft_mag;
}

void breath::update_buf(const cv::Mat& val, cv::Mat &buf, int &len, bool cumsum_flag, bool overlapadd_flag) {
    if (val.empty()) return;

    if (buf.empty()) {
        val.copyTo(buf);
    } else {
        if (!cumsum_flag) {
            if (buf.cols < len) {
                cv::hconcat(buf, val.col(val.cols - 1), buf);
            } else {
                cv::hconcat(buf.colRange(1, buf.cols), val.col(val.cols - 1), buf);

                if (overlapadd_flag) {
                    cv::add(buf.colRange(buf.cols - val.cols, buf.cols), val,
                            buf.colRange(buf.cols - val.cols, buf.cols));
                    cv::divide(buf.colRange(buf.cols - val.cols, buf.cols), 2,
                               buf.colRange(buf.cols - val.cols, buf.cols));
                }
            }
        } else {
            if (buf.cols < len) {
                cv::hconcat(buf, buf.col(buf.cols - 1) + val.col(val.cols - 1), buf); // cumsum
            } else {
                cv::hconcat(buf.colRange(1, buf.cols), buf.col(buf.cols - 1) + val.col(val.cols - 1),
                            buf); // cumsum
            }
        }
    }
}

void breath::modify_breathshape(cv::Mat &trace) {
    if (trace.empty()) return;

    float a = 4; //2;
    cv::Mat shape;
    cv::pow(trace, 2, shape);
    cv::divide(1 + (a * a), 1 + (a * a) * shape, shape);
    cv::pow(shape, 0.5, shape);
    cv::multiply(trace, shape, trace);
}

void breath::gen_bbox(const int &frame_width, const int &frame_height, std::vector <cv::Rect> &bbox) {
    std::vector<int> x_range = std::vector < int > {0, frame_width};
    std::vector<int> y_range = std::vector < int > {0, frame_height};

    for (int i = 0; i < param.bbox_size.size(); i++) {
        for (int x = x_range.at(0);
             x <= x_range.at(1) - param.bbox_size.at(i); x = x +
                                                             floor(param.bbox_size.at(i) * param.bbox_step.at(i))) {
            for (int y = y_range.at(0); y <= y_range.at(1) - param.bbox_size.at(i); y = y +
                                                                                        floor(param.bbox_size.at(
                                                                                                i) *
                                                                                              param.bbox_step.at(
                                                                                                      i))) {
                bbox.push_back(cv::Rect(x, y, param.bbox_size.at(i), param.bbox_size.at(i)));
            }
        }
    }
}

cv::Mat breath::createA(const int &framelen, const float &lambda) {
    cv::Mat D = cv::Mat::zeros(framelen - 2, framelen, CV_32FC1);

    for (int y = 0; y < D.rows; y++) {
        for (int x = 0; x < D.cols; x++) {
            if (y == x) D.at<float>(y, x) = 1;
            if (y == x - 1) D.at<float>(y, x) = -2;
            if (y == x - 2) D.at<float>(y, x) = 1;
        }
    }

    cv::Mat I = cv::Mat::eye(framelen, framelen, CV_32FC1);

    cv::Mat DD;
    cv::gemm(D, D, (float) (lambda * lambda), cv::Mat(), 0, DD, cv::GEMM_1_T);

    cv::Mat IDD;
    cv::add(I, DD, IDD);

    cv::Mat A = IDD.inv(cv::DECOMP_SVD); // cv::subtract(I, IDD.inv(cv::DECOMP_SVD), A);

    return A;
}

cv::Mat breath::createC(const int &framelen, const float &lambda_low, const float &lambda_hig) {
    cv::Mat A_low = createA(framelen, lambda_low);
    cv::Mat A_hig = createA(framelen, lambda_hig);

    cv::Mat I = cv::Mat::eye(framelen, framelen, CV_32FC1);

    cv::Mat C;
    cv::gemm(A_low, I - A_hig, 1, cv::Mat(), 0, C); // C = B * (eye(L) - A);
    return C;
}

cv::Mat breath::create_hann(int &size) {
    cv::Mat hann = cv::Mat(1, size, CV_32FC1);

    for (int i = 0; i < size; i++) {
        hann.at<float>(0, i) = 0.5 * (1 - cos((float) 2 * 3.14159265358979323846f * i / ((float) size - 1)));
    }

    return hann;
}

void breath::concatenate(const cv::Mat &trace, std::vector<float> &breath_signal, const int &len) {
    if (trace.empty()) return;

    if (breath_signal.empty()) {
        breath_signal.assign((float *) trace.datastart, (float *) trace.dataend);
    } else {
        breath_signal.push_back(trace.at<float>(0, trace.cols - 1));
    }

    if (breath_signal.size() > len) {
        breath_signal.erase(breath_signal.begin());
    }
}

void breath::concatenate(const float &activity, std::vector<float> &activity_signal, const int &len) {
    activity_signal.push_back(activity);

    if (activity_signal.size() > len) {
        activity_signal.erase(activity_signal.begin());
    }
}

void breath::reset() {
    localtrace_shift_dx = cv::Mat();
    localtrace_shift_dy = cv::Mat();
    localtrace_resp = cv::Mat();
    localtrace_acti = cv::Mat();
    globaltrace_resprate = cv::Mat();
    globaltrace_activity = cv::Mat();
    breath_trace_ref = cv::Mat();
    motion = 0;
}