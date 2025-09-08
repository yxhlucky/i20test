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
    param.activity_thr_hig = 3; // activity threshold high
    param.qual_thr = 5; // 5;
    param.chestroi_thr = 0.8;
    param.chest_present_thr = fps * 60 * 2; // 2 min

    param.breath_signal_len = 1 * 60 * fps; // 1 min buffer
    param.activity_signal_len = 1 * 60 * fps; // 1 min buffer
    param.breathrate_len = 1 * 60 * fps; // 1 min buffer

    vote_reset_counter = 0;
    vote_reset_interval = fps * 30;
    std::vector<std::deque<cv::Mat>> region_imbuf(4);

    region_votes = std::vector<float>(4, 0.0f);

}

breath_info breath::detect(const cv::Mat &rgb) {

    // update frame buffer
    update_frame_buffer(rgb, imbuf);


    //   auto t0 = std::chrono::high_resolution_clock::now();
    // generate local shifts 2-3ms
    std::vector<cv::Mat> shifts = gen_xy_shifts(imbuf);
//    auto t1 = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double, std::milli> ms = t1 - t0;
//    std::cout << "[Timer] gen_xy_shifts took " << ms.count() << " ms\n";

    // generate breath trace  1.5-2.5ms
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
    bi.bbox_select =  param.bbox;
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

//old

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


//new

/*

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
            float sum_dxdt = idxdt.at<float>(y2 + 1, x2 + 1) + idxdt.at<float>(y1, x1) - idxdt.at<float>(y2 + 1, x1) -
                             idxdt.at<float>(y1, x2 + 1);
            float sum_dt = idt.at<float>(y2 + 1, x2 + 1) + idt.at<float>(y1, x1) - idt.at<float>(y2 + 1, x1) -
                           idt.at<float>(y1, x2 + 1);

            float ux = sum_dxdt;

            float uy = sum_dydt;

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

//cv::Mat combineXY3(const cv::Mat &trace_dx, const cv::Mat &trace_dy) {
//    static cv::Mat trace_dx_avg, trace_dy_avg;
//    cv::reduce(trace_dx, trace_dx_avg, 1, cv::REDUCE_AVG);
//    cv::reduce(trace_dy, trace_dy_avg, 1, cv::REDUCE_AVG);
//
//    static cv::Mat trace_dx_avg_mat, trace_dy_avg_mat;
//    cv::repeat(trace_dx_avg, 1, trace_dx.cols, trace_dx_avg_mat);
//    cv::repeat(trace_dy_avg, 1, trace_dy.cols, trace_dy_avg_mat);
//
//    static cv::Mat trace_dx_n, trace_dy_n;
//    cv::subtract(trace_dx, trace_dx_avg_mat, trace_dx_n);
//    cv::subtract(trace_dy, trace_dy_avg_mat, trace_dy_n);
//
//    static cv::Mat trace_dx_pow, trace_dy_pow;
//    cv::pow(trace_dx_n, 2, trace_dx_pow);
//    cv::pow(trace_dy_n, 2, trace_dy_pow);
//
//    static cv::Mat trace_dx_std, trace_dy_std;
//    cv::reduce(trace_dx_pow, trace_dx_std, 1, cv::REDUCE_AVG);
//    cv::sqrt(trace_dx_std, trace_dx_std);
//    cv::reduce(trace_dy_pow, trace_dy_std, 1, cv::REDUCE_AVG);
//    cv::sqrt(trace_dy_std, trace_dy_std);
//
//    cv::Mat norm_sq;
//    cv::add(trace_dx_std.mul(trace_dx_std), trace_dy_std.mul(trace_dy_std), norm_sq); //避免 pow 和 sqrt
//    cv::Mat norm;
//    cv::sqrt(norm_sq, norm);
//    norm += 1e-6;
//    cv::Mat Wx = trace_dx_std / norm; // 广播
//    cv::Mat Wy = trace_dy_std / norm; // 广播
//
//    static cv::Mat trace_dxdy_prod;
//    cv::multiply(trace_dx_n, trace_dy_n, trace_dxdy_prod);
//
//    static cv::Mat trace_dxdy_prod_sum;
//    cv::reduce(trace_dxdy_prod, trace_dxdy_prod_sum, 1, cv::REDUCE_SUM);
//
//    static cv::Mat mask;
//    cv::compare(trace_dxdy_prod_sum, 0, mask, cv::CMP_LT);
//
//    cv::Mat sign = cv::Mat::ones(mask.rows, 1, CV_32FC1);
//    sign.setTo(-1, mask);
//
//    cv::multiply(Wy, sign, Wy);
//
//    static cv::Mat Wy2, WxWy, Wx2;
//    cv::multiply(Wx, Wx, Wx2);
//    cv::multiply(Wy, Wy, Wy2);
//    cv::multiply(Wx, Wy, WxWy);
//
//    static cv::Mat a, b;
//    cv::add(Wx2, WxWy, a);
//    cv::add(Wy2, WxWy, b);
//
//    static cv::Mat a_mat, b_mat;
//    cv::repeat(a, 1, trace_dx.cols, a_mat);
//    cv::repeat(b, 1, trace_dy.cols, b_mat);
//
//    cv::multiply(a_mat, trace_dx_n, a_mat);
//    cv::multiply(b_mat, trace_dy_n, b_mat);
//
//    cv::Mat S;
//    cv::subtract(trace_dy, trace_dy_avg_mat, trace_dy_n);
//    trace_dy_n.copyTo(S);
//    return S;
//
////    cv::Mat S;
////    cv::add(a_mat, b_mat, S);
////
////    return S;
//}

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

//原始breath
//cv::Mat breath::gen_breath_trace(const cv::Mat &dx, const cv::Mat &dy) {
//    update_buf(dx, localtrace_shift_dx, param.motion_len, true, false);
//    update_buf(dy, localtrace_shift_dy, param.motion_len, true, false);
//
//    // detrend and denoise
//    if (localtrace_shift_dx.cols == param.motion_len) {
//        // Option 3: STD mat based
//        cv::Mat localtrace_shift = combineXY3(localtrace_shift_dx, localtrace_shift_dy);
//
//        // remove non-respiratory components
//        detrend(localtrace_shift, param.detrend_mat);
//
//        update_buf(localtrace_shift, localtrace_resp, param.localtrace_resp_len, false, true);
//    }
//
//    if (localtrace_resp.cols != param.localtrace_resp_len) {
//        return cv::Mat();
//    }
//
//    // normalize local signals
//    cv::Mat localtrace_sel_n = remove_avg_std(localtrace_resp, 1);
//
//    // correct sign
//    cv::Mat trace_ac;
//    if (!breath_trace_ref.empty()) {
//        mxu::Mat mxu_In_Mat_localtrace_sel_n(localtrace_sel_n);
//        mxu::_InputArray mxu_In_Arr_localtrace_sel_n(mxu_In_Mat_localtrace_sel_n);
//        mxu::Mat mxu_In_Mat_breath_trace_ref(breath_trace_ref);
//        mxu::_InputArray mxu_In_Arr_breath_trace_ref(mxu_In_Mat_breath_trace_ref);
//        mxu::Mat mxu_Out_Mat_trace_ac(trace_ac);
//        mxu::_OutputArray mxu_Out_Arr_trace_ac(mxu_Out_Mat_trace_ac);
//        mxu::gemm(mxu_In_Arr_localtrace_sel_n, mxu_In_Arr_breath_trace_ref, 1, mxu::Mat(), 0, mxu_Out_Arr_trace_ac,
//                  mxu::GEMM_2_T);
//        mxu_Out_Mat_trace_ac = mxu_Out_Arr_trace_ac.getMat();
//        trace_ac = mxu_Out_Mat_trace_ac.toCvMat();
//        //cv::gemm(localtrace_sel_n, breath_trace_ref, 1, cv::Mat(), 0, trace_ac, cv::GEMM_2_T);
//    } else {
//        trace_ac = cv::Mat::ones(localtrace_sel_n.rows, 1, CV_32FC1);
//    }
//
//    cv::normalize(trace_ac, softmask, 1, 0, cv::NORM_L2);
//
//    // generate breath trace
//    cv::Mat breath_trace;
//    mxu::Mat mxu_In_Mat_softmask(softmask);
//    mxu::_InputArray mxu_In_Arr_softmask(mxu_In_Mat_softmask);
//    mxu::Mat mxu_In_Mat_localtrace_sel_n(localtrace_sel_n);
//    mxu::_InputArray mxu_In_Arr_localtrace_sel_n(mxu_In_Mat_localtrace_sel_n);
//
//    mxu::Mat mxu_Out_Mat_breath_trace(breath_trace);
//    mxu::_OutputArray mxu_Out_Arr_breath_trace(mxu_Out_Mat_breath_trace);
//    mxu::gemm(mxu_In_Arr_softmask, mxu_In_Arr_localtrace_sel_n, 1, mxu::Mat(), 0, mxu_Out_Arr_breath_trace,
//              mxu::GEMM_1_T);
//    mxu_Out_Mat_breath_trace = mxu_Out_Arr_breath_trace.getMat();
//    breath_trace = mxu_Out_Mat_breath_trace.toCvMat();
//
//    //cv::gemm(softmask, localtrace_sel_n, 1, cv::Mat(), 0, breath_trace, cv::GEMM_1_T);
//
//    cv::Scalar avg, stddev;
//    cv::meanStdDev(breath_trace, avg, stddev);
//    breath_trace = (breath_trace - avg[0]) / (1 + stddev[0]);
//
//    // update reference breath
//    cv::hconcat(breath_trace.colRange(1, breath_trace.cols), cv::Mat::zeros(1, 1, CV_32FC1), breath_trace_ref);
//
//    return breath_trace;
//}


/**********/
/*

cv::Mat breath::gen_breath_trace(const cv::Mat &dx, const cv::Mat &dy) {
    // 增加计数器
    vote_reset_counter++;

    update_buf(dx, localtrace_shift_dx, param.motion_len, true, false);
    update_buf(dy, localtrace_shift_dy, param.motion_len, true, false);

    // 在localtrace_resp填满后进行处理
    if (localtrace_shift_dx.cols == param.motion_len) {
        // 处理短窗口数据
        cv::Mat localtrace_shift = combineXY3(localtrace_shift_dx, localtrace_shift_dy);
        detrend(localtrace_shift, param.detrend_mat);
        update_buf(localtrace_shift, localtrace_resp, param.localtrace_resp_len, false, true);
    }

    // 当15秒长窗口准备好后进行区域投票

    if (localtrace_resp.cols != param.localtrace_resp_len) {
        return cv::Mat();
    }
    if (localtrace_resp.rows != static_cast<int>(bbox_region.size())) {
        std::cerr << "[ERROR] localtrace_resp.rows (" << localtrace_resp.rows
                  << ") != bbox_region.size() (" << bbox_region.size() << ")\n";
        // 让程序停下来，便于 gdb 定位：
        CV_Assert(localtrace_resp.rows == static_cast<int>(bbox_region.size()));
    }


    cv::Mat signal_energy = cv::Mat::zeros(localtrace_resp.rows, 1, CV_32FC1);
    for (int i = 0; i < localtrace_resp.rows; ++i) {
        cv::Mat row = localtrace_resp.row(i);
        cv::Mat fft;
        cv::dft(row, fft, cv::DFT_COMPLEX_OUTPUT);
        std::vector<cv::Mat> planes;
        cv::split(fft, planes);
        cv::Mat mag;
        cv::magnitude(planes[0], planes[1], mag);
        mag = mag(cv::Range(0, mag.rows), cv::Range(0, mag.cols / 2 + 1));
        double max_val, sum_val;
        cv::minMaxLoc(mag, nullptr, &max_val);
        sum_val = cv::sum(mag)[0];
        signal_energy.at<float>(i, 0) = static_cast<float>(max_val / (sum_val - max_val + 1e-6));
    }

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

    // 选出最佳区域
    best_region = std::max_element(std::begin(slide_sum), std::end(slide_sum)) - std::begin(slide_sum);
                 // 已处理帧计数
    region_votes_900[best_region]++;        // 记录当前帧最佳区域
    frame_900++;

    if (frame_900 >= 900) {                 // 每 900 帧（30 秒）判定一次
        int winner = std::max_element(region_votes_900, region_votes_900+4) - region_votes_900;
        int max_vote = region_votes_900[winner];
        ratio = (float)max_vote / 900.0f;
        printf("[900-frame] 区域票数 = [%d, %d, %d, %d]  最高占比 = %.2f  "
               "结论 = %s\n",
               region_votes_900[0], region_votes_900[1],
               region_votes_900[2], region_votes_900[3], ratio,
               ratio >= 0.7f ? "REAL" : "FAKE");

        memset(region_votes_900, 0, sizeof(region_votes_900));
        frame_900 = 0;
    }


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

/*
//8区域选1
cv::Mat breath::gen_breath_trace(const cv::Mat &dx, const cv::Mat &dy) {
    // 增加计数器
    vote_reset_counter++;

    update_buf(dx, localtrace_shift_dx, param.motion_len, true, false);
    update_buf(dy, localtrace_shift_dy, param.motion_len, true, false);

    // 在localtrace_resp填满后进行处理
    if (localtrace_shift_dx.cols == param.motion_len) {
        // 处理短窗口数据
        cv::Mat localtrace_shift = combineXY3(localtrace_shift_dx, localtrace_shift_dy);
        detrend(localtrace_shift, param.detrend_mat);
        update_buf(localtrace_shift, localtrace_resp, param.localtrace_resp_len, false, true);
    }

    // 当15秒长窗口准备好后进行区域投票

    if (localtrace_resp.cols != param.localtrace_resp_len) {
        return cv::Mat();
    }
    if (localtrace_resp.rows != static_cast<int>(bbox_region.size())) {
        std::cerr << "[ERROR] localtrace_resp.rows (" << localtrace_resp.rows
                  << ") != bbox_region.size() (" << bbox_region.size() << ")\n";
        // 让程序停下来，便于 gdb 定位：
        CV_Assert(localtrace_resp.rows == static_cast<int>(bbox_region.size()));
    }


    cv::Mat signal_energy = cv::Mat::zeros(localtrace_resp.rows, 1, CV_32FC1);
    for (int i = 0; i < localtrace_resp.rows; ++i) {
        cv::Mat row = localtrace_resp.row(i);
        cv::Mat fft;
        cv::dft(row, fft, cv::DFT_COMPLEX_OUTPUT);
        std::vector<cv::Mat> planes;
        cv::split(fft, planes);
        cv::Mat mag;
        cv::magnitude(planes[0], planes[1], mag);
        mag = mag(cv::Range(0, mag.rows), cv::Range(0, mag.cols / 2 + 1));
        double max_val, sum_val;
        cv::minMaxLoc(mag, nullptr, &max_val);
        sum_val = cv::sum(mag)[0];
        signal_energy.at<float>(i, 0) = static_cast<float>(max_val / (sum_val - max_val + 1e-6));
    }

    assert(best_region >= 0 && best_region < 8);
    assert(slide_idx >= 0 && slide_idx < SLIDE_LEN);

    // 滑动窗口投票
    for (int r = 0; r < 8; ++r) {
        slide_sum[r] -= slide_buf[r][slide_idx];
    }

    for (size_t i = 0; i < bbox_region.size(); ++i) {
        int r = bbox_region[i];
        if (r < 0 || r >= 8) continue;

        float val = signal_energy.at<float>(i);
        slide_buf[r][slide_idx] = val;
        slide_sum[r] += val;
    }
    slide_idx = (slide_idx + 1) % SLIDE_LEN;

    // 选出最佳区域
    best_region = std::max_element(std::begin(slide_sum), std::end(slide_sum)) - std::begin(slide_sum);
    // 已处理帧计数
    region_votes_900[best_region]++;        // 记录当前帧最佳区域
    frame_900++;

    if (frame_900 >= 900) {                 // 每 900 帧（30 秒）判定一次
        int winner = std::max_element(region_votes_900, region_votes_900+8) - region_votes_900;
        int max_vote = region_votes_900[winner];
        ratio = (float)max_vote / 900.0f;
        printf("[1800-frame] 区域票数 = [%d, %d, %d, %d, %d, %d, %d, %d]  最高占比 = %.2f  结论 = %s\n",
               region_votes_900[0], region_votes_900[1], region_votes_900[2], region_votes_900[3],
               region_votes_900[4], region_votes_900[5], region_votes_900[6], region_votes_900[7],
               ratio, ratio >= 0.6f ? "REAL" : "FAKE");



        memset(region_votes_900, 0, sizeof(region_votes_900));
        frame_900 = 0;
    }


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

//8区域选2
/*

cv::Mat breath::gen_breath_trace(const cv::Mat &dx, const cv::Mat &dy)
{
    // 计数器自增
    vote_reset_counter++;

    update_buf(dx, localtrace_shift_dx, param.motion_len, true, false);
    update_buf(dy, localtrace_shift_dy, param.motion_len, true, false);

    // 短窗口长度满足后进行处理
    if (localtrace_shift_dx.cols == param.motion_len)
    {
        cv::Mat localtrace_shift = combineXY3(localtrace_shift_dx, localtrace_shift_dy);
        detrend(localtrace_shift, param.detrend_mat);
        update_buf(localtrace_shift, localtrace_resp, param.localtrace_resp_len, false, true);
    }

    // 长窗口未就绪则提前返回
    if (localtrace_resp.cols != param.localtrace_resp_len)
        return cv::Mat();

    // 行数必须与 bbox_region 数量一致
    if (localtrace_resp.rows != static_cast<int>(bbox_region.size()))
    {
        std::cerr << "[ERROR] localtrace_resp.rows (" << localtrace_resp.rows
                  << ") != bbox_region.size() (" << bbox_region.size() << ")\n";
        CV_Assert(localtrace_resp.rows == static_cast<int>(bbox_region.size()));
    }

    // 计算每个区域信号能量
    cv::Mat signal_energy = cv::Mat::zeros(localtrace_resp.rows, 1, CV_32FC1);
    for (int i = 0; i < localtrace_resp.rows; ++i)
    {
        cv::Mat fft;
        cv::dft(localtrace_resp.row(i), fft, cv::DFT_COMPLEX_OUTPUT);
        std::vector<cv::Mat> planes;
        cv::split(fft, planes);
        cv::Mat mag;
        cv::magnitude(planes[0], planes[1], mag);
        mag = mag(cv::Range(0, mag.rows), cv::Range(0, mag.cols / 2 + 1));

        double max_val, sum_val;
        cv::minMaxLoc(mag, nullptr, &max_val);
        sum_val = cv::sum(mag)[0];
        signal_energy.at<float>(i) = static_cast<float>(max_val / (sum_val - max_val + 1e-6f));
    }

    assert(best_region >= 0 && best_region < 8);
    assert(slide_idx >= 0 && slide_idx < SLIDE_LEN);

    // 滑动窗口投票：先减旧值
    for (int r = 0; r < 8; ++r)
        slide_sum[r] -= slide_buf[r][slide_idx];

    // 加入新值
    for (size_t i = 0; i < bbox_region.size(); ++i)
    {
        int r = bbox_region[i];
        if (r < 0 || r >= 8) continue;
        float val = signal_energy.at<float>(i);
        slide_buf[r][slide_idx] = val;
        slide_sum[r] += val;
    }
    slide_idx = (slide_idx + 1) % SLIDE_LEN;

    // 选出当前最佳区域
    best_region = std::max_element(slide_sum, slide_sum + 8) - slide_sum;

    // 取能量最高的前两个区域
    std::vector<std::pair<int, float>> region_pairs;
    region_pairs.reserve(8);
    for (int i = 0; i < 8; ++i)
        region_pairs.emplace_back(i, slide_sum[i]);

    std::sort(region_pairs.begin(), region_pairs.end(),
              [](const std::pair<int, float> &a, const std::pair<int, float> &b)
              { return a.second > b.second; });

    std::vector<int> top2_regions;
    for (int i = 0, top_k = std::min(2, static_cast<int>(region_pairs.size())); i < top_k; ++i)
    {
        if (region_pairs[i].second > 0.0f)
            top2_regions.push_back(region_pairs[i].first);
    }
// ===== 新增：用两位十进制整数编码 region_code =====
    if (top2_regions.empty())
    {
        region_code = -1;
    }
    else if (top2_regions.size() == 1)
    {
        region_code = top2_regions[0] * 10 + top2_regions[0];   // 例如 3 → 33
    }
    else
    {
        region_code = top2_regions[0] * 10 + top2_regions[1];   // 例如 0 和 7 → 07
    }
    //std::cout<<" region = "<<region_code<<std::endl;
    // =====================================================
    // 900 帧统计
    region_votes_900[best_region]++;
    frame_900++;
    if (frame_900 >= 900)
    {
        int winner = std::max_element(region_votes_900, region_votes_900 + 8) - region_votes_900;
        int max_vote = region_votes_900[winner];
        ratio = static_cast<float>(max_vote) / 900.0f;
        printf("[1800-frame] 区域票数 = [%d,%d,%d,%d,%d,%d,%d,%d]  最高占比=%.2f  结论=%s\n",
               region_votes_900[0], region_votes_900[1], region_votes_900[2], region_votes_900[3],
               region_votes_900[4], region_votes_900[5], region_votes_900[6], region_votes_900[7],
               ratio, ratio >= 0.6f ? "REAL" : "FAKE");
        memset(region_votes_900, 0, sizeof(region_votes_900));
        frame_900 = 0;
    }

    // 构造 mask
    cv::Mat mask = cv::Mat::zeros(localtrace_resp.rows, 1, CV_32FC1);
    int box_count = 0;
    for (size_t i = 0; i < bbox_region.size(); ++i)
    {
        if (std::find(top2_regions.begin(), top2_regions.end(), bbox_region[i]) != top2_regions.end())
        {
            mask.at<float>(i) = 1.0f;
            box_count++;
        }
    }
    if (box_count == 0)
    {
        mask = cv::Mat::ones(localtrace_resp.rows, 1, CV_32FC1);
        box_count = mask.rows;
    }

    // 加权平均得到呼吸信号
    cv::Mat breath_trace = cv::Mat::zeros(1, localtrace_resp.cols, CV_32FC1);
    const float inv_box = 1.0f / box_count;
    for (int i = 0; i < localtrace_resp.rows; ++i)
    {
        if (mask.at<float>(i) > 0.0f)
            breath_trace += localtrace_resp.row(i) * inv_box;
    }

    // 归一化
    cv::Scalar avg, stddev;
    cv::meanStdDev(breath_trace, avg, stddev);
    breath_trace = (breath_trace - avg[0]) / (1.0 + stddev[0]);

    // 生成参考 trace（供下一帧差分使用）
    if (breath_trace.cols > 1)
    {
        cv::hconcat(breath_trace.colRange(1, breath_trace.cols),
                    cv::Mat::zeros(1, 1, CV_32FC1),
                    breath_trace_ref);
    }
    return breath_trace;
}

*/
cv::Mat breath::gen_breath_trace(const cv::Mat &dx, const cv::Mat &dy)
{
    /* ---------- 1. 缓存与预处理 ---------- */
    update_buf(dx, localtrace_shift_dx, param.motion_len, true, false);
    update_buf(dy, localtrace_shift_dy, param.motion_len, true, false);

    if (localtrace_shift_dx.cols == param.motion_len) {
        cv::Mat localtrace_shift = combineXY3(localtrace_shift_dx, localtrace_shift_dy);
        detrend(localtrace_shift, param.detrend_mat);
        update_buf(localtrace_shift, localtrace_resp, param.localtrace_resp_len, false, true);
    }

    if (localtrace_resp.cols != param.localtrace_resp_len)
        return cv::Mat();

    cv::Mat signal_energy = calc_fft_max(localtrace_resp);

    /* ---------- 2. 多区域投票 ---------- */
    /* 清空本轮要写的一整列 */
    for (int r = 0; r < 8; ++r) {
        slide_sum[r] -= slide_buf[r][slide_idx];
        slide_buf[r][slide_idx] = 0.f;
    }
    /* 每个 box 只往自己所属区域累加一次 */
    for (size_t i = 0; i < bbox_regions.size(); ++i) {
        float val = signal_energy.at<float>(i);
        for (int r : bbox_regions[i]) {
            slide_buf[r][slide_idx] += val;
            slide_sum[r] += val;
        }
    }
    slide_idx = (slide_idx + 1) % SLIDE_LEN;

    /* ---------- 3. 1800 帧真人/假人判定 ---------- */
    static int region_votes_900[8] = {0};
    static int frame_900 = 0;

    best_region = std::max_element(slide_sum, slide_sum + 8) - slide_sum;

    region_votes_900[best_region]++;
    if (++frame_900 >= 900) {
        int winner = std::max_element(region_votes_900, region_votes_900 + 8) - region_votes_900;
        float ratio = region_votes_900[winner] / 1800.0f;
        printf("[1800-frame] 区域票数=[%d,%d,%d,%d,%d,%d,%d,%d] 最高占比=%.2f 结论=%s\n",
               region_votes_900[0], region_votes_900[1], region_votes_900[2], region_votes_900[3],
               region_votes_900[4], region_votes_900[5], region_votes_900[6], region_votes_900[7],
               ratio, ratio >= 0.7f ? "REAL" : "FAKE");
        memset(region_votes_900, 0, sizeof(region_votes_900));
        frame_900 = 0;
    }

    /* ---------- 4. 构建呼吸信号（仅 best_region 的 box） ---------- */
    cv::Mat mask = cv::Mat::zeros(localtrace_resp.rows, 1, CV_32FC1);
    int box_count = 0;
    for (size_t i = 0; i < bbox_regions.size(); ++i) {
        bool take = false;
        for (int r : bbox_regions[i]) if (r == best_region) { take = true; break; }
        if (take) {
            mask.at<float>(i, 0) = 1.0f;
            ++box_count;
        }
    }
    if (box_count == 0) {
        mask = cv::Mat::ones(localtrace_resp.rows, 1, CV_32FC1);
        box_count = mask.rows;
    }

    cv::Mat breath_trace = cv::Mat::zeros(1, localtrace_resp.cols, CV_32FC1);
    for (int i = 0; i < localtrace_resp.rows; ++i)
        if (mask.at<float>(i, 0) > 0)
            breath_trace += localtrace_resp.row(i) / box_count;

    cv::Scalar avg, stddev;
    cv::meanStdDev(breath_trace, avg, stddev);
    breath_trace = (breath_trace - avg[0]) / (1.f + stddev[0]);

    if (breath_trace.cols > 1) {
        cv::hconcat(breath_trace.colRange(1, breath_trace.cols),
                    cv::Mat::zeros(1, 1, CV_32FC1),
                    breath_trace_ref);
    }
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

void breath::detrend(cv::Mat &trace, const cv::Mat A) {
    cv::gemm(trace, A, 1, cv::Mat(), 0, trace);
}


int breath::calc_breathrate(const cv::Mat &breath_trace, const float &activity) {
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
    if (breathrate >= param.min_breathrate && breathrate <= param.max_breathrate) // || quality > qual_thr)
    {
        if (activity > param.activity_thr_low && activity < param.activity_thr_hig)
            update_buf(cv::Mat(1, 1, CV_32FC1, (float) breathrate), globaltrace_resprate,
                       param.globaltrace_resprate_len, false, false);

        if (globaltrace_resprate.empty()) {
            breathrate = 0;
            return breathrate;
        }
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

cv::Mat breath::gen_spectrum(const cv::Mat &trace) {
    cv::Mat fft, fft_mag;
    cv::Mat tmp[2] = {trace, cv::Mat::zeros(trace.size(), trace.type())};
    cv::merge(tmp, 2, fft);
    cv::dft(fft, fft, cv::DFT_SCALE + cv::DFT_COMPLEX_OUTPUT + cv::DFT_ROWS);

    cv::split(fft.colRange(0, floor(fft.cols / 2) - 1), tmp);
    cv::magnitude(tmp[0], tmp[1], fft_mag);

    return fft_mag;
}

void breath::update_buf(const cv::Mat &val, cv::Mat &buf, int &len, bool cumsum_flag, bool overlapadd_flag) {
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
//四区域
/*

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

*/
/*

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
        float center_x = bbox[i].x + bbox[i].width / 2.0f;
        float center_y = bbox[i].y + bbox[i].height / 2.0f;

        // 分配区域 (九宫格中的八个区域，不包括中心区域)
        if (center_y < h_third) {
            if (center_x < w_third) {
                bbox_region[i] = 0; // 左上区域
            } else if (center_x >= 2 * w_third) {
                bbox_region[i] = 2; // 右上区域
            } else {
                bbox_region[i] = 1; // 上方区域
            }
        } else if (center_y >= 2 * h_third) {
            if (center_x < w_third) {
                bbox_region[i] = 5; // 左下区域
            } else if (center_x >= 2 * w_third) {
                bbox_region[i] = 7; // 右下区域
            } else {
                bbox_region[i] = 6; // 下方区域
            }
        } else {
            if (center_x < w_third) {
                bbox_region[i] = 3; // 左侧区域
            } else if (center_x >= 2 * w_third) {
                bbox_region[i] = 4; // 右侧区域
            } // 中心区域不分配
        }
    }

    // 添加在函数末尾 - 打印区域分配结果
    std::cout << "==== Box Region Assignment Results ====" << std::endl;
    std::cout << "Total boxes: " << bbox.size() << std::endl;

    // 区域计数器
    int region_counts[9] = {0}; // 0-7为八个区域，8用于未分配区域(-1)

    for (size_t i = 0; i < bbox.size(); i++) {
        std::cout << "Box " << i << ": (" << bbox[i].x << ", " << bbox[i].y
                  << ", " << bbox[i].width << ", " << bbox[i].height << ") ";

        std::cout << "Center: (" << (bbox[i].x + bbox[i].width / 2.0f) << ", "
                  << (bbox[i].y + bbox[i].height / 2.0f) << ") ";

        switch (bbox_region[i]) {
            case 0:
                std::cout << "Region: TOP-LEFT" << std::endl;
                region_counts[0]++;
                break;
            case 1:
                std::cout << "Region: TOP-RIGHT" << std::endl;
                region_counts[1]++;
                break;
            case 2:
                std::cout << "Region: TOP" << std::endl;
                region_counts[2]++;
                break;
            case 3:
                std::cout << "Region: BOTTOM-LEFT" << std::endl;
                region_counts[3]++;
                break;
            case 4:
                std::cout << "Region: BOTTOM-RIGHT" << std::endl;
                region_counts[4]++;
                break;
            case 5:
                std::cout << "Region: BOTTOM" << std::endl;
                region_counts[5]++;
                break;
            case 6:
                std::cout << "Region: LEFT" << std::endl;
                region_counts[6]++;
                break;
            case 7:
                std::cout << "Region: RIGHT" << std::endl;
                region_counts[7]++;
                break;
            default:
                std::cout << "Region: UNASSIGNED" << std::endl;
                region_counts[8]++;
                break;
        }
    }

    // 打印统计信息
    std::cout << "\n==== Region Statistics ====" << std::endl;
    std::cout << "TOP-LEFT region: " << region_counts[0] << " boxes" << std::endl;
    std::cout << "TOP-RIGHT region: " << region_counts[1] << " boxes" << std::endl;
    std::cout << "TOP region: " << region_counts[2] << " boxes" << std::endl;
    std::cout << "BOTTOM-LEFT region: " << region_counts[3] << " boxes" << std::endl;
    std::cout << "BOTTOM-RIGHT region: " << region_counts[4] << " boxes" << std::endl;
    std::cout << "BOTTOM region: " << region_counts[5] << " boxes" << std::endl;
    std::cout << "LEFT region: " << region_counts[6] << " boxes" << std::endl;
    std::cout << "RIGHT region: " << region_counts[7] << " boxes" << std::endl;
    std::cout << "UNASSIGNED: " << region_counts[8] << " boxes" << std::endl;
    std::cout << "================================" << std::endl;

}

*/
void breath::gen_bbox(const int &frame_width, const int &frame_height, std::vector <cv::Rect> &bbox) {
    /* ========== 1. 原有滑动窗口生成逻辑（不变） ========== */
    std::vector<int> x_range{0, frame_width};
    std::vector<int> y_range{0, frame_height};

    for (size_t i = 0; i < param.bbox_size.size(); ++i)
    {
        int bs  = param.bbox_size.at(i);
        int st  = static_cast<int>(std::floor(bs * param.bbox_step.at(i)));
        for (int y = y_range[0]; y <= y_range[1] - bs; y += st)
            for (int x = x_range[0]; x <= x_range[1] - bs; x += st)
                bbox.emplace_back(x, y, bs, bs);
    }
    x_range.clear();
    y_range.clear();

    /* ========== 2. 固定 8 个 40×40 区域（仅打印用） ========== */
    const int win = 40;
    const std::vector<cv::Rect> fixed8 = {
            {0,   0,   win, win},   // 0
            {20,  0,   win, win},   // 1
            {40,  0,   win, win},   // 2
            {0,   20,  win, win},   // 3
            {40,  20,  win, win},   // 4
            {0,   40,  win, win},   // 5
            {20,  40,  win, win},   // 6
            {40,  40,  win, win}    // 7
    };

    /* ========== 3. 手工映射：box 索引 → 所属区域列表 ========== */
    /* 80×80 画面，20×20 步长 20 → 4×4 = 16 个 box */
    /* ========== 3. 行优先 4×4 映射表 ========== */
    const std::vector<std::vector<int>> box_regions = {
            {0},             // 0  (0,0)
            {0,1},           // 1  (0,20)
            {1,2},           // 2  (0,40)
            {2},             // 3  (0,60)
            {0,3},           // 4  (20,0)
            {0,1,3},         // 5  (20,20)
            {1,2,4},         // 6  (20,40)
            {2,4},           // 7  (20,60)
            {3,5},           // 8  (40,0)
            {3,5,6},         // 9  (40,20)
            {4,6,7},         // 10 (40,40)
            {4,7},           // 11 (40,60)
            {5},             // 12 (60,0)
            {5,6},           // 13 (60,20)
            {6,7},           // 14 (60,40)
            {7}              // 15 (60,60)
    };

    /* ========== 4. 计算每个 box 所属区域 ========== */
    bbox_regions.resize(bbox.size());

    for (size_t i = 0; i < bbox.size() && i < box_regions.size(); ++i)
        bbox_regions[i] = box_regions[i];

    /* ========== 5. 打印结果 ========== */
    int region_counts[8] = {0};

    std::cout << "==== Box Region Assignment Results ====" << std::endl;
    std::cout << "Total boxes: " << bbox.size() << std::endl;

    for (size_t i = 0; i < bbox.size(); ++i)
    {
        std::cout << "Box " << i << ": (" << bbox[i].x << "," << bbox[i].y
                  << ") regions:";
        for (int r : bbox_regions[i]) std::cout << " " << r;
        std::cout << std::endl;

        for (int r : bbox_regions[i]) ++region_counts[r];
    }

    std::cout << "\n==== Region Statistics ====" << std::endl;
    for (int r = 0; r < 8; ++r)
        std::cout << "Region " << r << ": " << region_counts[r] << " boxes" << std::endl;
    std::cout << "================================" << std::endl;
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