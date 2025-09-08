#include "breath.h"

breath::breath(const cv::Size frame_size, const int fps) {
    param.fps = fps;
    param.frame_size = frame_size;

    gen_bbox(param.frame_size.width, param.frame_size.height, param.bbox);

    //gen_accumap();
    param.motion_len = 3 * fps; // 3 sec length
    param.localtrace_resp_len = 15 * fps; // 15 sec length
    param.localtrace_acti_len = 10 * fps; // 10 sec length

    param.detrend_mat = createC(param.motion_len, 1, 100);
    param.hann = create_hann(param.localtrace_resp_len);

    param.globaltrace_resprate_len = fps * 3; // 3-sec buffer
    param.globaltrace_activity_len = fps * 1; // 3-sec buffer

    param.activity_thr_low = 0.1; // activity threshold low
    param.activity_thr_hig = 5; // activity threshold high
    param.breathrate_minmax_thr = 10; // 5 bpm contrast
    param.qual_thr = 5; // 5;
    param.chestroi_thr = 0.8;
    param.chest_present_thr = fps * 60 * 2; // 2 min






    vote_reset_counter = 0;
    vote_reset_interval = fps * 30;
    std::vector<std::deque<cv::Mat>> region_imbuf(4);
    // 初始化投票数组
    region_votes = std::vector<float>(4, 0.0f);
}

breath_info breath::detect(const cv::Mat &rgb, const cv::Rect chestroi) {
    if (rgb.empty())
    {
        breath_info empty_bi;
        // 可选：把关键字段置空/置 0，方便外部识别
        empty_bi.breath_signal.clear();
        empty_bi.breath_rates.clear();
        empty_bi.activity_signal.clear();
        return empty_bi;
    }
    // update frame buffer
    update_frame_buffer(rgb, imbuf);

    // generate local shifts
    std::vector <cv::Mat> shifts = gen_xy_shifts(imbuf, chestroi);

    // generate breath trace
    cv::Mat breath_trace = gen_breath_trace(shifts.at(0), shifts.at(1), chestroi);

    // generate activity
    float activity = gen_activity_trace(shifts.at(2));

    // generate breath rate
    int breathrate = calc_breathrate(breath_trace, activity);

    modify_breathshape(breath_trace);

    // temporal accumulate
    concatenate(breath_trace, breath_signal, 1 * 60 * param.fps); // generate breath signal
    concatenate(activity, activity_signal, 1 * 60 * param.fps); // generat activity signal
    concatenate((float) breathrate, breath_rates, 1 * 60 * param.fps); // generat activity signal

    // set breath_info
    breath_info bi;
    bi.breath_signal = breath_signal;
    bi.breath_rates = breath_rates;
    bi.activity_signal = activity_signal;
    bi.chestroi = chestroi;
    bi.bbox_select = param.bbox;
    return bi;
}

cv::Rect breath::update_chestroi(const cv::Mat frame, const cv::Rect faceroi) {
    cv::Rect chestroi;

    if (!faceroi.empty()) {
        //chestroi = cv::Rect(faceroi.x - faceroi.width * 1, faceroi.y - faceroi.height * 1, faceroi.width * 3, faceroi.height * 3);
        chestroi = cv::Rect(faceroi.x - faceroi.width * 1, faceroi.y + faceroi.height * 1, faceroi.width * 3,
                            faceroi.height * 1);
        chestroi = chestroi & cv::Rect(0, 0, frame.cols, frame.rows);
    }

    return chestroi;
}

void breath::update_frame_buffer(const cv::Mat frame, std::deque <cv::Mat> &imbuf) {
    if (frame.empty()) return;

    cv::Mat gray, grayf;

    // check frame channel
    if (frame.channels() > 1) {
        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);
        gray.convertTo(grayf, CV_32FC1, (float) 1. / 255); // convert uint8 to float32, 255 to 1
    } else {
        frame.convertTo(grayf, CV_32FC1, (float) 1. / 255); // convert uint8 to float32, 255 to 1
    }
    // normalize local intensity
    // cv::divide(grayf, cv::mean(grayf), grayf);

    imbuf.push_back(grayf);
    if (imbuf.size() == 4) // N-1 frames buffer
    {
        imbuf.pop_front();
    }
}
void breath::overlap_add(const cv::Mat val, cv::Mat &buf, int len) {
    if (val.empty()) return;

    if (buf.empty()) {
        val.copyTo(buf);
    } else {
        temp_buf.create(buf.rows, buf.cols + 1, buf.type());
        cv::hconcat(buf, cv::Mat::zeros(1, 1, CV_32FC1), temp_buf);
        buf = temp_buf;

        if (buf.cols > len) {
            buf.colRange(1, buf.cols).copyTo(buf);
        }
        cv::add(buf.colRange(buf.cols - val.cols, buf.cols), val, buf.colRange(buf.cols - val.cols, buf.cols));
    }
}
//old
/*

std::vector <cv::Mat> breath::gen_xy_shifts(const std::deque <cv::Mat> imbuf, const cv::Rect roi) {
    if (imbuf.size() < 2) {
        //cv::Mat localda = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdx = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdy = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdt = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);

        return std::vector < cv::Mat > {localdx, localdy, localdt};
    } else {
        cv::Mat frame_sub = imbuf.at(0) - imbuf.at(imbuf.size() - 1);
        cv::Mat frame_add = (imbuf.at(0) + imbuf.at(imbuf.size() - 1)) / 2;

        cv::Mat dx, dy, dt;
        cv::filter2D(frame_add, dx, CV_32FC1, param.kernel_dx, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(frame_add, dy, CV_32FC1, param.kernel_dy, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
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
        cv::Mat localdx = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdy = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdt = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);

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

*/


//new

std::vector <cv::Mat> breath::gen_xy_shifts(const std::deque <cv::Mat> imbuf, const cv::Rect roi) {
    if (imbuf.size() < 2) {
        //cv::Mat localda = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdx = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdy = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdt = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);

        return std::vector < cv::Mat > {localdx, localdy, localdt};
    } else {
        cv::Mat frame_sub = imbuf.at(0) - imbuf.at(imbuf.size() - 1);
        cv::Mat frame_add = (imbuf.at(0) + imbuf.at(imbuf.size() - 1)) / 2;

        cv::Mat dx, dy, dt;
        cv::filter2D(frame_add, dx, CV_32FC1, param.kernel_dx, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::filter2D(frame_add, dy, CV_32FC1, param.kernel_dy, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
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
        cv::Mat localdx = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdy = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdt = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);

        for (int i = 0; i < param.bbox.size(); i++) {
            int x1 = param.bbox.at(i).x;
            int x2 = param.bbox.at(i).x + param.bbox.at(i).width - 1;
            int y1 = param.bbox.at(i).y;
            int y2 = param.bbox.at(i).y + param.bbox.at(i).height - 1;

            float sum_dydt = idydt.at<float>(y2 + 1, x2 + 1) + idydt.at<float>(y1, x1) - idydt.at<float>(y2 + 1, x1) -
                             idydt.at<float>(y1, x2 + 1);
            float sum_dxdt = idxdt.at<float>(y2 + 1, x2 + 1) + idxdt.at<float>(y1, x1) - idxdt.at<float>(y2 + 1, x1) -
                             idxdt.at<float>(y1, x2 + 1);
            float sum_dt = idt.at<float>(y2 + 1, x2 + 1) + idt.at<float>(y1, x1) - idt.at<float>(y2 + 1, x1) -
                           idt.at<float>(y1, x2 + 1);

            float ux = sum_dxdt;
            float uy = sum_dydt;
            localdx.at<float>(i, 0) = ux;
            localdy.at<float>(i, 0) = uy;
            localdt.at<float>(i, 0) = 100 * sum_dt / param.bbox.at(i).area();
        }
        //localshift_mag = localshift_mag / (1e-3 + cv::norm(localshift_mag, cv::NORM_L2));
        //cv::multiply(localshift_ang, localshift_mag, localshift_ang);

        return std::vector < cv::Mat > {localdx, localdy, localdt};
    }
}






/*cv::Mat combineXY3(const cv::Mat trace_dx, const cv::Mat trace_dy) {
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

    static cv::Mat trace_dx_std_pow, trace_dy_std_pow;
    cv::pow(trace_dx_std, 2, trace_dx_std_pow);
    cv::pow(trace_dy_std, 2, trace_dy_std_pow);

    static cv::Mat norm;
    cv::add(trace_dx_std_pow, trace_dy_std_pow, norm);
    cv::sqrt(norm, norm);

    static cv::Mat Wx, Wy;
    norm += 1e-6;
    cv::divide(trace_dx_std, norm, Wx);
    cv::divide(trace_dy_std, norm, Wy);

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
   // cv::subtract(trace_dy, trace_dy_avg_mat, trace_dy_n);
    trace_dy_n.copyTo(S);
//    return S;
//    cv::Mat S;
//    cv::subtract(trace_dx, trace_dx_avg_mat, trace_dx_n);
//    trace_dx_n.copyTo(S);
//    return S;
//    cv::Mat S;
//    cv::add(a_mat, b_mat, S);

    return S;
}*/



// 与 MATLAB v3 版本完全对齐
// resp : 每行是一条信号，列 = 帧数
// 返回 : 每行的“信噪比” e，尺寸 (rows, 1) CV_32F
cv::Mat calc_fft_max(const cv::Mat& resp) {
    CV_Assert(resp.type() == CV_32F && resp.dims == 2);

    int rows = resp.rows;
    int cols = resp.cols;

    // 1) FFT
    cv::Mat planes[] = { resp.clone(), cv::Mat::zeros(resp.size(), CV_32F) };
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);

    cv::dft(complexI, complexI, cv::DFT_ROWS | cv::DFT_SCALE);

    // 2) 取幅值
    cv::split(complexI, planes);        // planes[0] = Re, planes[1] = Im
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat mag = planes[0];            // 幅值矩阵 (rows, cols)

    // 3) 只保留后半部分频谱（正频率）
    int half = cols / 2;
    cv::Mat magHalf = mag(cv::Range::all(), cv::Range(half, cols));

    // 4) 计算 e = max / (sum - max)
    cv::Mat maxVal, sumVal;
    cv::reduce(magHalf, maxVal, 1, cv::REDUCE_MAX);
    cv::reduce(magHalf, sumVal, 1, cv::REDUCE_SUM);

    cv::Mat denom;
    cv::subtract(sumVal, maxVal, denom);
    cv::divide(maxVal, denom, maxVal);   // e = max / (sum - max)

    return maxVal;   // (rows, 1) CV_32F
}

/*

cv::Mat combineXY3(const cv::Mat& trace_dx, const cv::Mat& trace_dy) {
    CV_Assert(trace_dx.type() == CV_32F && trace_dy.type() == CV_32F);
    CV_Assert(trace_dx.size() == trace_dy.size());

    int rows = trace_dx.rows;
    int cols = trace_dx.cols;

    // 1. 中心化
    cv::Mat dx_centered = trace_dx.clone();
    cv::Mat dy_centered = trace_dy.clone();

    for (int i = 0; i < rows; ++i) {
        float dx_mean = cv::mean(trace_dx.row(i))[0];
        float dy_mean = cv::mean(trace_dy.row(i))[0];

        dx_centered.row(i) = trace_dx.row(i) - dx_mean;
        dy_centered.row(i) = trace_dy.row(i) - dy_mean;
    }

    // 2. FFT 选择方向
    cv::Mat x_energy = calc_fft_max(dx_centered);
    cv::Mat y_energy = calc_fft_max(dy_centered);

    cv::Mat combXY(rows, cols, CV_32F);

    cv::Mat S(dx_centered.rows, dx_centered.cols, dx_centered.type());


    for (int i = 0; i < rows; ++i) {
        float x_score = x_energy.at<float>(i, 0);
        float y_score = y_energy.at<float>(i, 0);

        if (x_score > y_score) {
            dx_centered.row(i).copyTo(combXY.row(i));

        } else {
            dy_centered.row(i).copyTo(combXY.row(i));

        }
    }

    return combXY;
}
*/

//能量


cv::Mat combineXY3(const cv::Mat& trace_dx, const cv::Mat& trace_dy) {
    CV_Assert(trace_dx.type() == CV_32F && trace_dy.type() == CV_32F);
    CV_Assert(trace_dx.size() == trace_dy.size());

    // 1. 中心化（去均值）
    static cv::Mat trace_dx_avg, trace_dy_avg;
    cv::reduce(trace_dx, trace_dx_avg, 1, cv::REDUCE_AVG);
    cv::reduce(trace_dy, trace_dy_avg, 1, cv::REDUCE_AVG);


    static cv::Mat trace_dx_avg_mat, trace_dy_avg_mat;
    cv::repeat(trace_dx_avg, 1, trace_dx.cols, trace_dx_avg_mat);
    cv::repeat(trace_dy_avg, 1, trace_dy.cols, trace_dy_avg_mat);

    static cv::Mat trace_dx_n, trace_dy_n;
    cv::subtract(trace_dx, trace_dx_avg_mat, trace_dx_n);
    cv::subtract(trace_dy, trace_dy_avg_mat, trace_dy_n);


    cv::Mat x_energy = calc_fft_max(trace_dx_n);
    cv::Mat y_energy = calc_fft_max(trace_dy_n);

    // 3. 构造输出矩阵
    cv::Mat S(trace_dx_n.rows, trace_dx_n.cols, trace_dx_n.type());

    for (int i = 0; i < trace_dx_n.rows; ++i) {
        float x_score = x_energy.at<float>(i, 0);
        float y_score = y_energy.at<float>(i, 0);

        if (x_score > y_score) {
            cv::Mat row = trace_dx_n.row(i).clone();
            double dot = row.dot(trace_dy_n.row(i));
            if (dot < 0) {
                row *= -1;
            }
            row.copyTo(S.row(i));
        } else {
            trace_dy_n.row(i).copyTo(S.row(i));
        }
    }

    return S;
}


//std
/*

cv::Mat combineXY3(const cv::Mat trace_dx, const cv::Mat trace_dy) {
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


    cv::Mat S(trace_dx_n.rows, trace_dx_n.cols, trace_dx_n.type());


    for (int i = 0; i < trace_dx_n.rows; i++) {
        float x_std_val = trace_dx_std.at<float>(i, 0);
        float y_std_val = trace_dy_std.at<float>(i, 0);

        if (x_std_val > y_std_val) {
            cv::Mat row = trace_dx_n.row(i).clone();
            double dot = row.dot(trace_dy_n.row(i));
            if (dot < 0) {
                row *= -1;
            }
            row.copyTo(S.row(i));
            //trace_dx_n.row(i).copyTo(S.row(i));
        } else {
            trace_dy_n.row(i).copyTo(S.row(i));
        }
    }


    return S;
}

*/




cv::Mat remove_avg_std(const cv::Mat trace, float bias) {
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


/*
cv::Mat breath::gen_breath_trace(const cv::Mat dx, const cv::Mat dy, const cv::Rect chestroi) {
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

    // refine softmask
    //cv::Mat softmask;
    cv::normalize(trace_ac, softmask, 1, 0, cv::NORM_L2);
    // softmask = refine_mask(softmask);

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
*/






//std
/*

cv::Mat breath::gen_breath_trace(const cv::Mat dx, const cv::Mat dy, const cv::Rect chestroi) {
    // 增加计数器
    vote_reset_counter++;

    update_buf(dx, localtrace_shift_dx, param.motion_len, true, false);
    update_buf(dy, localtrace_shift_dy, param.motion_len, true, false);

    // 在localtrace_resp填满后进行处理
    if (localtrace_shift_dx.cols == param.motion_len) {
        // 处理短窗口数据
        cv::Mat localtrace_shift = combineXY3(localtrace_shift_dx, localtrace_shift_dy);
        //detrend(localtrace_shift, param.detrend_mat);
        update_buf(localtrace_shift, localtrace_resp, param.localtrace_resp_len, false, true);
    }

    // 当15秒长窗口准备好后进行区域投票

    if (localtrace_resp.cols != param.localtrace_resp_len) {
        return cv::Mat();
    }

    // 1. 直接计算原始信号的标准差作为投票依据
    cv::Mat signal_std = cv::Mat::zeros(localtrace_resp.rows, 1, CV_32FC1);
    for (int i = 0; i < localtrace_resp.rows; i++) {
        cv::Scalar mean, stddev;
        cv::meanStdDev(localtrace_resp.row(i), mean, stddev);
        signal_std.at<float>(i, 0) = stddev[0];
    }
    assert(best_region >= 0 && best_region < 4);
    assert(slide_idx >= 0 && slide_idx < SLIDE_LEN);
    // 先把旧票减掉
    for (int r = 0; r < 4; ++r) {
        slide_sum[r] -= slide_buf[r][slide_idx];
    }

    // 再压入本轮新票
    for (size_t i = 0; i < bbox_region.size(); ++i) {
        int r = bbox_region[i];
        if (r < 0 || r >= 4) continue;

        float val = signal_std.at<float>(i);
        slide_buf[r][slide_idx] = val;
        slide_sum[r] += val;
    }

    // 指针前进
    slide_idx = (slide_idx + 1) % SLIDE_LEN;

    //-------------------------------------------------
    // 5. 选出最佳区域
    //-------------------------------------------------
    best_region = std::max_element(std::begin(slide_sum), std::end(slide_sum)) -
                  std::begin(slide_sum);


//    best_region = 2;
    cv::Mat visualization = cv::Mat::zeros(chestroi.height, chestroi.width, CV_8UC3);

    // 定义区域颜色 (BGR格式) - 使用较暗的颜色
    cv::Scalar colors[5] = {
            cv::Scalar(0, 100, 150),   // 上方区域 (暗橙色)
            cv::Scalar(0, 150, 0),     // 左侧区域 (暗绿色)
            cv::Scalar(150, 0, 0),     // 下方区域 (暗蓝色)
            cv::Scalar(0, 0, 150),     // 右侧区域 (暗红色)
            cv::Scalar(50, 50, 50)     // 未分配区域 (暗灰色)
    };

    // 选中区域的高亮颜色 (更亮)
    cv::Scalar highlight_colors[4] = {
            cv::Scalar(0, 200, 255),   // 上方区域 (亮橙色)
            cv::Scalar(0, 255, 0),     // 左侧区域 (亮绿色)
            cv::Scalar(255, 0, 0),     // 下方区域 (亮蓝色)
            cv::Scalar(0, 0, 255)      // 右侧区域 (亮红色)
    };

    // 绘制所有box - 使用param.bbox而不是bbox
    for (size_t i = 0; i < param.bbox.size(); i++) {
        // 选择颜色 - 高亮显示最佳区域的box
        cv::Scalar color;
        if (i < bbox_region.size() && bbox_region[i] == best_region) {
            color = highlight_colors[best_region]; // 高亮颜色
            cv::rectangle(visualization, param.bbox[i], color, -1); // 填充矩形
        } else if (i < bbox_region.size() && bbox_region[i] >= 0 && bbox_region[i] < 4) {
            color = colors[bbox_region[i]];
            cv::rectangle(visualization, param.bbox[i], color, 1); // 只绘制边界
        } else {
            color = colors[4]; // 灰色表示未分配区域
            cv::rectangle(visualization, param.bbox[i], color, 1); // 只绘制边界
        }
    }


    // 显示可视化图像
    cv::imshow("呼吸区域选择", visualization);
    cv::waitKey(1); // 更新显示，不等待按键

    // 打印每个区域的票数和总积分
//    std::cout << "区域投票结果 (当前票数):" << std::endl;
//    std::string region_names[4] = {"上方区域", "左侧区域", "下方区域", "右侧区域"};
//    for (int i = 0; i < 4; ++i) {
//        std::cout << region_names[i] << ": " << slide_sum[i] << std::endl;
//    }
//    std::cout << "最佳区域: " << region_names[best_region] << std::endl;
//    std::cout << "----------------------------------------" << std::endl;

    // 5. 创建mask，只保留最佳区域的boxes
    cv::Mat mask = cv::Mat::zeros(localtrace_resp.rows, 1, CV_32FC1);
    int box_count = 0;

    for (size_t i = 0; i < bbox_region.size(); i++) {
        if (bbox_region[i] == best_region) {
            mask.at<float>(i, 0) = 1.0f;
            box_count++;
        }
    }

    // 如果没有有效区域，使用所有box
    if (box_count == 0) {
        mask = cv::Mat::ones(localtrace_resp.rows, 1, CV_32FC1);
        box_count = mask.rows;
    }

    // 6. 计算最佳区域的平均信号
    cv::Mat breath_trace = cv::Mat::zeros(1, localtrace_resp.cols, CV_32FC1);
    for (int i = 0; i < localtrace_resp.rows; i++) {
        if (mask.at<float>(i, 0) > 0) {
            breath_trace += localtrace_resp.row(i) / box_count;
        }
    }

    // 7. 对选出的信号进行归一化
    cv::Scalar avg, stddev;
    cv::meanStdDev(breath_trace, avg, stddev);
    breath_trace = (breath_trace - avg[0]) / (1 + stddev[0]);


    if (breath_trace.cols > 1) {
        cv::hconcat(breath_trace.colRange(1, breath_trace.cols), cv::Mat::zeros(1, 1, CV_32FC1), breath_trace_ref);
    }

    return breath_trace;
}

*/

//能量


/*
cv::Mat breath::gen_breath_trace(const cv::Mat dx, const cv::Mat dy, const cv::Rect chestroi) {
    vote_reset_counter++;

    update_buf(dx, localtrace_shift_dx, param.motion_len, true, false);
    update_buf(dy, localtrace_shift_dy, param.motion_len, true, false);

    if (localtrace_shift_dx.cols == param.motion_len) {
        cv::Mat localtrace_shift = combineXY3(localtrace_shift_dx, localtrace_shift_dy);
       // detrend(localtrace_shift, param.detrend_mat);
        update_buf(localtrace_shift, localtrace_resp, param.localtrace_resp_len, false, true);
    }

    if (localtrace_resp.cols != param.localtrace_resp_len) {
        return cv::Mat();
    }

    cv::Mat signal_energy = calc_fft_max(localtrace_resp);

    assert(best_region >= 0 && best_region < 4);
    assert(slide_idx >= 0 && slide_idx < SLIDE_LEN);

    // 滑动窗口投票
    for (int r = 0; r < 4; ++r) {
        slide_sum[r] -= slide_buf[r][slide_idx];
    }

    for (size_t i = 0; i < bbox_region.size(); ++i) {
        int r = bbox_region[i];
        if (r < 0 || r >= 4) continue;

        float val = signal_energy.at<float>(i);
        slide_buf[r][slide_idx] = val;
        slide_sum[r] += val;
    }

    slide_idx = (slide_idx + 1) % SLIDE_LEN;
    printf("[Vote] slide_sum: [0]=%.3f  [1]=%.3f  [2]=%.3f  [3]=%.3f\n",
           slide_sum[0], slide_sum[1], slide_sum[2], slide_sum[3]);
    // 选出最佳区域
    best_region = std::max_element(std::begin(slide_sum), std::end(slide_sum)) - std::begin(slide_sum);

    // 可视化
    cv::Mat visualization = cv::Mat::zeros(chestroi.height, chestroi.width, CV_8UC3);
    cv::Scalar colors[5] = {
            cv::Scalar(0, 100, 150),
            cv::Scalar(0, 150, 0),
            cv::Scalar(150, 0, 0),
            cv::Scalar(0, 0, 150),
            cv::Scalar(50, 50, 50)
    };
    cv::Scalar highlight_colors[4] = {
            cv::Scalar(0, 200, 255),
            cv::Scalar(0, 255, 0),
            cv::Scalar(255, 0, 0),
            cv::Scalar(0, 0, 255)
    };

    for (size_t i = 0; i < param.bbox.size(); ++i) {
        cv::Scalar color;
        if (i < bbox_region.size() && bbox_region[i] == best_region) {
            color = highlight_colors[best_region];
            cv::rectangle(visualization, param.bbox[i], color, -1);
        } else if (i < bbox_region.size() && bbox_region[i] >= 0 && bbox_region[i] < 4) {
            color = colors[bbox_region[i]];
            cv::rectangle(visualization, param.bbox[i], color, 1);
        } else {
            color = colors[4];
            cv::rectangle(visualization, param.bbox[i], color, 1);
        }
    }

    cv::imshow("呼吸区域选择", visualization);
    cv::waitKey(1);

    // 创建 mask
    cv::Mat mask = cv::Mat::zeros(localtrace_resp.rows, 1, CV_32FC1);
    int box_count = 0;
    for (size_t i = 0; i < bbox_region.size(); ++i) {
        if (bbox_region[i] == best_region) {
            mask.at<float>(i, 0) = 1.0f;
            box_count++;
        }
    }

    if (box_count == 0) {
        mask = cv::Mat::ones(localtrace_resp.rows, 1, CV_32FC1);
        box_count = mask.rows;
    }

    // 计算平均信号
    cv::Mat breath_trace = cv::Mat::zeros(1, localtrace_resp.cols, CV_32FC1);
    for (int i = 0; i < localtrace_resp.rows; ++i) {
        if (mask.at<float>(i, 0) > 0) {
            breath_trace += localtrace_resp.row(i) / box_count;
        }
    }

    // 归一化
    cv::Scalar avg, stddev;
    cv::meanStdDev(breath_trace, avg, stddev);
    breath_trace = (breath_trace - avg[0]) / (1 + stddev[0]);

    if (breath_trace.cols > 1) {
        cv::hconcat(breath_trace.colRange(1, breath_trace.cols), cv::Mat::zeros(1, 1, CV_32FC1), breath_trace_ref);
    }

    return breath_trace;
}
*/
cv::Mat breath::gen_breath_trace(const cv::Mat dx, const cv::Mat dy, const cv::Rect chestroi) {
    vote_reset_counter++;

    update_buf(dx, localtrace_shift_dx, param.motion_len, true, false);
    update_buf(dy, localtrace_shift_dy, param.motion_len, true, false);

    if (localtrace_shift_dx.cols == param.motion_len) {
        cv::Mat localtrace_shift = combineXY3(localtrace_shift_dx, localtrace_shift_dy);
        detrend(localtrace_shift, param.detrend_mat);
        update_buf(localtrace_shift, localtrace_resp, param.localtrace_resp_len, false, true);
    }

    if (localtrace_resp.cols != param.localtrace_resp_len) {
        return cv::Mat();
    }

    cv::Mat signal_energy = calc_fft_max(localtrace_resp);

    assert(best_region >= 0 && best_region < 4);
    assert(slide_idx >= 0 && slide_idx < SLIDE_LEN);

    /* ---------- 滑动窗口投票 ---------- */
    for (int r = 0; r < 4; ++r) {
        slide_sum[r] -= slide_buf[r][slide_idx];
    }

    for (size_t i = 0; i < bbox_region.size(); ++i) {
        int r = bbox_region[i];
        if (r < 0 || r >= 4) continue;

        float val = signal_energy.at<float>(i);
        slide_buf[r][slide_idx] = val;
        slide_sum[r] += val;
    }
    slide_idx = (slide_idx + 1) % SLIDE_LEN;

    /* ---------- 实时投票柱状图（新增） ---------- */
    {
        // 1) 归一化到 0~1
        float max_sum = *std::max_element(slide_sum, slide_sum + 4);
        max_sum = std::max(max_sum, 1e-6f);
        const int VOTE_H = 150;
        const int VOTE_W = 400;

        // 2) 画布：只在第一次创建
        static cv::Mat vote_canvas;
        if (vote_canvas.empty()) {
            vote_canvas = cv::Mat(VOTE_H, VOTE_W, CV_8UC3, cv::Scalar(30, 30, 30));
            cv::namedWindow("Vote-4Region", cv::WINDOW_AUTOSIZE);
        }
        vote_canvas.setTo(cv::Scalar(30, 30, 30));

        // 3) 画柱子
        int bar_w = VOTE_W / 4 - 10;
        cv::Scalar bar_colors[4] = {
                cv::Scalar(0, 200, 255),   // 0
                cv::Scalar(0, 255, 0),     // 1
                cv::Scalar(255, 0, 0),     // 2
                cv::Scalar(0, 0, 255)      // 3
        };
        for (int i = 0; i < 4; ++i) {
            int h = static_cast<int>((slide_sum[i] / max_sum) * (VOTE_H - 20));
            cv::Rect r(i * (bar_w + 10) + 10, VOTE_H - h - 10, bar_w, h);
            cv::rectangle(vote_canvas, r, bar_colors[i], -1);

            // 数值
            cv::putText(vote_canvas,
                        std::to_string(int(slide_sum[i])),
                        cv::Point(r.x, VOTE_H - 5),
                        cv::FONT_HERSHEY_PLAIN,
                        1.0,
                        cv::Scalar(255, 255, 255),
                        1);
        }

        // 4) 高亮最佳区域
        int x = best_region * (bar_w + 10) + 10;
        cv::Rect hl(x - 2, VOTE_H - 2, bar_w + 4, 2);
        cv::rectangle(vote_canvas, hl, cv::Scalar(255, 255, 255), -1);

        cv::imshow("Vote-4Region", vote_canvas);
    }

    /* ---------- 原有可视化（胸部 ROI） ---------- */
    best_region = std::max_element(std::begin(slide_sum), std::end(slide_sum)) - std::begin(slide_sum);

    cv::Mat visualization = cv::Mat::zeros(chestroi.height, chestroi.width, CV_8UC3);
    cv::Scalar colors[5] = {
            cv::Scalar(0, 100, 150),
            cv::Scalar(0, 150, 0),
            cv::Scalar(150, 0, 0),
            cv::Scalar(0, 0, 150),
            cv::Scalar(50, 50, 50)
    };
    cv::Scalar highlight_colors[4] = {
            cv::Scalar(0, 200, 255),
            cv::Scalar(0, 255, 0),
            cv::Scalar(255, 0, 0),
            cv::Scalar(0, 0, 255)
    };

    for (size_t i = 0; i < param.bbox.size(); ++i) {
        cv::Scalar color;
        if (i < bbox_region.size() && bbox_region[i] == best_region) {
            color = highlight_colors[best_region];
            cv::rectangle(visualization, param.bbox[i], color, -1);
        } else if (i < bbox_region.size() && bbox_region[i] >= 0 && bbox_region[i] < 4) {
            color = colors[bbox_region[i]];
            cv::rectangle(visualization, param.bbox[i], color, 1);
        } else {
            color = colors[4];
            cv::rectangle(visualization, param.bbox[i], color, 1);
        }
    }
    cv::imshow("呼吸区域选择", visualization);
    cv::waitKey(1);

    /* ---------- 后续逻辑保持不变 ---------- */

    /* ---------- 900 帧真人/假人判定 ---------- */
    static int region_votes_900[4] = {0};   // 900 帧内每个区域的当选次数
    static int frame_900 = 0;               // 已处理帧计数

    region_votes_900[best_region]++;        // 记录当前帧最佳区域
    frame_900++;

    static bool is_human = true;            // 上一周期结论，默认真人

    if (frame_900 >= 1800) {                 // 每 900 帧（30 秒）判定一次
        int winner = std::max_element(region_votes_900, region_votes_900+4) - region_votes_900;
        int max_vote = region_votes_900[winner];
        float ratio = (float)max_vote / 900.0f;

        printf("[900-frame] 区域票数 = [%d, %d, %d, %d]  最高占比 = %.2f  "
               "结论 = %s\n",
               region_votes_900[0], region_votes_900[1],
               region_votes_900[2], region_votes_900[3], ratio,
               ratio >= 0.7f ? "REAL" : "FAKE");

        /* 可视化判定结果 */
        const int FLAG_W = 220, FLAG_H = 100;
        static cv::Mat flag_canvas;
        if (flag_canvas.empty()) {
            flag_canvas = cv::Mat(FLAG_H, FLAG_W, CV_8UC3);
            cv::namedWindow("900-Frame Judge", cv::WINDOW_AUTOSIZE);
        }
        is_human = (ratio >= 0.7f);
        flag_canvas.setTo(is_human ? cv::Scalar(0,255,0) : cv::Scalar(0,0,255));
        cv::putText(flag_canvas, is_human ? "REAL" : "FAKE",
                    cv::Point(20, FLAG_H-25), cv::FONT_HERSHEY_SIMPLEX,
                    1.5, cv::Scalar(255,255,255), 3);
        cv::imshow("900-Frame Judge", flag_canvas);

        /* 清零准备下一轮 900 帧 */
        memset(region_votes_900, 0, sizeof(region_votes_900));
        frame_900 = 0;
    }

    /* ---------- 若判定为假人，直接返回空信号 ---------- */


    cv::Mat mask = cv::Mat::zeros(localtrace_resp.rows, 1, CV_32FC1);
    int box_count = 0;
    for (size_t i = 0; i < bbox_region.size(); ++i) {
        if (bbox_region[i] == best_region) {
            mask.at<float>(i, 0) = 1.0f;
            box_count++;
        }
    }
    if (box_count == 0) {
        mask = cv::Mat::ones(localtrace_resp.rows, 1, CV_32FC1);
        box_count = mask.rows;
    }

    cv::Mat breath_trace = cv::Mat::zeros(1, localtrace_resp.cols, CV_32FC1);
    for (int i = 0; i < localtrace_resp.rows; ++i) {
        if (mask.at<float>(i, 0) > 0) {
            breath_trace += localtrace_resp.row(i) / box_count;
        }
    }

    cv::Scalar avg, stddev;
    cv::meanStdDev(breath_trace, avg, stddev);
    breath_trace = (breath_trace - avg[0]) / (1 + stddev[0]);

    if (breath_trace.cols > 1) {
        cv::hconcat(breath_trace.colRange(1, breath_trace.cols),
                    cv::Mat::zeros(1, 1, CV_32FC1),
                    breath_trace_ref);
    }
    return breath_trace;
}

float breath::gen_activity_trace(const cv::Mat localdt) {
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

    if (activity_tmp > 10) {
        activity_tmp = 10;
    }

    // temporal average
    update_buf(cv::Mat(1, 1, CV_32FC1, (float) activity_tmp), globaltrace_activity, param.globaltrace_activity_len,
               false, false);

    float activity = (float) cv::mean(globaltrace_activity).val[0];
    motion = (float) cv::mean(globaltrace_activity).val[0];
    return activity;
}

void breath::detrend(cv::Mat &trace, const cv::Mat A) {
    cv::gemm(trace, A, 1, cv::Mat(), 0, trace);
}

int breath::calc_breathrate(const cv::Mat breath_trace, const float activity) {
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

    // check and update buffer
    if (breathrate >= 5 && breathrate <= 100 && activity > param.activity_thr_low && activity < param.activity_thr_hig) // || quality > qual_thr)
    {
        update_buf(cv::Mat(1, 1, CV_32FC1, (float)breathrate), globaltrace_resprate, param.globaltrace_resprate_len, false, false);
    }

    if (activity <= param.activity_thr_low) // no-subject case
    {
        breathrate = 0;
        globaltrace_resprate = cv::Mat();
    }

    if (globaltrace_resprate.cols == param.globaltrace_resprate_len) {
        // check resprate buffer contrast (max-min)
        double min_breathrate = -1, max_breathrate = -1;
        cv::minMaxIdx(globaltrace_resprate, &min_breathrate, &max_breathrate);
        int breathrate_dif = (int) (max_breathrate - min_breathrate);

        // calculate average respiratory rate
        breathrate = breathrate_dif < param.breathrate_minmax_thr ? (int) cv::mean(globaltrace_resprate)[0] : 0;
    }

    return breathrate;
}

cv::Mat breath::gen_spectrum(const cv::Mat trace) {
    cv::Mat fft, fft_mag;
    cv::Mat tmp[2] = {trace, cv::Mat::zeros(trace.size(), trace.type())};
    cv::merge(tmp, 2, fft);
    cv::dft(fft, fft, cv::DFT_SCALE + cv::DFT_COMPLEX_OUTPUT + cv::DFT_ROWS);

    cv::split(fft.colRange(0, floor(fft.cols / 2) - 1), tmp);
    cv::magnitude(tmp[0], tmp[1], fft_mag);

    return fft_mag;
}

void breath::update_buf(const cv::Mat val, cv::Mat &buf, int len, bool cumsum_flag, bool overlapadd_flag) {
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
                cv::hconcat(buf.colRange(1, buf.cols), buf.col(buf.cols - 1) + val.col(val.cols - 1), buf); // cumsum
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

void breath::gen_bbox(const int frame_width, const int frame_height, std::vector <cv::Rect> &bbox) {
    std::vector<int> x_range = std::vector < int > {0, frame_width};
    std::vector<int> y_range = std::vector < int > {0, frame_height};

    for (int i = 0; i < param.bbox_size.size(); i++) {
        for (int x = x_range.at(0);
             x <= x_range.at(1) - param.bbox_size.at(i); x = x + floor(param.bbox_size.at(i) * param.bbox_step.at(i))) {
            for (int y = y_range.at(0); y <= y_range.at(1) - param.bbox_size.at(i); y = y +
                                                                                        floor(param.bbox_size.at(i) *
                                                                                              param.bbox_step.at(i))) {
                bbox.push_back(cv::Rect(x, y, param.bbox_size.at(i), param.bbox_size.at(i)));
            }
        }
    }

    x_range.clear();
    y_range.clear();
// 计算九宫格划分的边界
    float h_third = frame_height / 3.0f;
    float w_third = frame_width / 3.0f;

    // 为每个box分配区域
    bbox_region.clear();
    bbox_region.resize(bbox.size(), -1);

    for (size_t i = 0; i < bbox.size(); i++) {
        // 计算box中心点
        float center_x = bbox[i].x + bbox[i].width/2.0f;
        float center_y = bbox[i].y + bbox[i].height/2.0f;

        // 分配区域 (类似九宫格中的上、左、下、右四个区域)
        if (center_y < h_third && center_x >= w_third && center_x < 2*w_third) {
            bbox_region[i] = 0; // 上方区域
        } else if (center_x < w_third && center_y >= h_third && center_y < 2*h_third) {
            bbox_region[i] = 1; // 左侧区域
        } else if (center_y >= 2*h_third && center_x >= w_third && center_x < 2*w_third) {
            bbox_region[i] = 2; // 下方区域
        } else if (center_x >= 2*w_third && center_y >= h_third && center_y < 2*h_third) {
            bbox_region[i] = 3; // 右侧区域
        }
    }
    // 添加在函数末尾 - 打印区域分配结果
    std::cout << "==== Box Region Assignment Results ====" << std::endl;
    std::cout << "Total boxes: " << bbox.size() << std::endl;

    // 区域计数器
    int region_counts[5] = {0}; // 0-3为四个区域，4用于未分配区域(-1)

    for (size_t i = 0; i < bbox.size(); i++) {
        std::cout << "Box " << i << ": (" << bbox[i].x << ", " << bbox[i].y
                  << ", " << bbox[i].width << ", " << bbox[i].height << ") ";

        std::cout << "Center: (" << (bbox[i].x + bbox[i].width/2.0f) << ", "
                  << (bbox[i].y + bbox[i].height/2.0f) << ") ";

        if (bbox_region[i] == 0) {
            std::cout << "Region: TOP" << std::endl;
            region_counts[0]++;
        } else if (bbox_region[i] == 1) {
            std::cout << "Region: LEFT" << std::endl;
            region_counts[1]++;
        } else if (bbox_region[i] == 2) {
            std::cout << "Region: BOTTOM" << std::endl;
            region_counts[2]++;
        } else if (bbox_region[i] == 3) {
            std::cout << "Region: RIGHT" << std::endl;
            region_counts[3]++;
        } else {
            std::cout << "Region: UNASSIGNED" << std::endl;
            region_counts[4]++;
        }
    }

    // 打印统计信息
    std::cout << "\n==== Region Statistics ====" << std::endl;
    std::cout << "TOP region: " << region_counts[0] << " boxes" << std::endl;
    std::cout << "LEFT region: " << region_counts[1] << " boxes" << std::endl;
    std::cout << "BOTTOM region: " << region_counts[2] << " boxes" << std::endl;
    std::cout << "RIGHT region: " << region_counts[3] << " boxes" << std::endl;
    std::cout << "UNASSIGNED: " << region_counts[4] << " boxes" << std::endl;
    std::cout << "================================" << std::endl;


}

cv::Mat breath::createA(const int framelen, const float lambda) {
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

    // release local variables
    D.release();
    I.release();
    DD.release();
    IDD.release();

    return A;
}

cv::Mat breath::createC(const int framelen, const float lambda_low, const float lambda_hig) {
    cv::Mat A_low = createA(framelen, lambda_low);
    cv::Mat A_hig = createA(framelen, lambda_hig);

    cv::Mat I = cv::Mat::eye(framelen, framelen, CV_32FC1);

    cv::Mat C;
    cv::gemm(A_low, I - A_hig, 1, cv::Mat(), 0, C); // C = B * (eye(L) - A);

    I.release();
    A_low.release();
    A_hig.release();

    return C;
}

cv::Mat breath::create_hann(int size) {
    cv::Mat hann = cv::Mat(1, size, CV_32FC1);

    for (int i = 0; i < size; i++) {
        hann.at<float>(0, i) = 0.5 * (1 - cos((float) 2 * (float) (3.14159265358979323846) * i / ((float) size - 1)));
    }

    return hann;
}

void breath::concatenate(const cv::Mat trace, std::vector<float> &breath_signal, const int len) {
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

void breath::concatenate(const float activity, std::vector<float> &activity_signal, const int len) {
    activity_signal.push_back(activity);

    if (activity_signal.size() > len) {
        activity_signal.erase(activity_signal.begin());
    }
}


cv::Mat breath::refine_mask(const cv::Mat mask) {
    if (mask.empty()) {
        return cv::Mat();
    } else {
        //double st = (double)cv::getTickCount();
        cv::Mat heatmap = cv::Mat::zeros(param.frame_size, CV_32FC1);
        //cv::Mat accumap = cv::Mat::zeros(frame_size, CV_32FC1);
        for (int i = 0; i < param.bbox.size(); i++) {
            cv::add(heatmap(param.bbox.at(i)), cv::Scalar(mask.at<float>(i, 0)), heatmap(param.bbox.at(i)));
            //cv::add(accumap(bbox.at(i)), cv::Scalar(1), accumap(bbox.at(i)));
        }
        //double en = (double)cv::getTickCount();
        //double T = (en - st) / cv::getTickFrequency();
        //std::cout << "TTTTTTTTTTT = " << T << std::endl;

        cv::divide(heatmap, accumap_global + 1e-6, heatmap);
        //cv::divide(heatmap, accumap + 1e-6, heatmap);

        cv::Mat iheatmap;
        cv::integral(heatmap, iheatmap, CV_32FC1);

        cv::Mat mask_refine = cv::Mat::zeros(mask.size(), CV_32FC1);

        float *iheatmap_data = (float *) iheatmap.data;
        for (int i = 0; i < param.bbox.size(); i++) {
            int x1 = param.bbox.at(i).x;
            int x2 = param.bbox.at(i).x + param.bbox.at(i).width - 1;
            int y1 = param.bbox.at(i).y;
            int y2 = param.bbox.at(i).y + param.bbox.at(i).height - 1;

            mask_refine.at<float>(i, 0) =
                    iheatmap_data[(y2 + 1) * iheatmap.cols + (x2 + 1)] + iheatmap_data[(y1) * iheatmap.cols + (x1)]
                    - iheatmap_data[(y2 + 1) * iheatmap.cols + (x1)] - iheatmap_data[(y1) * iheatmap.cols + (x2 + 1)];
        }

        /*
        cv::Mat mask_refine = cv::Mat::zeros(mask.size(), CV_32FC1);
        for (int i = 0; i < this->bbox.size(); i++)
        {
            mask_refine.at<float>(i, 0) = (float) cv::mean(heatmap(bbox.at(i))).val[0];
        }
        */

        heatmap.release();
        //accumap.release();
        iheatmap.release();

        return mask_refine;
    }
}

void breath::gen_accumap() {
    accumap_global = cv::Mat::zeros(param.frame_size, CV_32FC1);

    for (int i = 0; i < param.bbox.size(); i++) {
        cv::add(accumap_global(param.bbox.at(i)), cv::Scalar(1), accumap_global(param.bbox.at(i)));
    }
}

