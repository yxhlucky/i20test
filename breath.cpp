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
}

breath_info breath::detect(const cv::Mat &rgb, const cv::Rect chestroi) {
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

        // check RoI
        /*
        if (!roi.empty())
        {
            cv::Mat mask = cv::Mat::zeros(dt.size(), CV_32FC1);
            mask(roi & cv::Rect(0, 0, dx.cols, dy.rows)).setTo(1);
            mask(faceroi2 & cv::Rect(0, 0, dx.cols, dy.rows)).setTo(0);
            cv::multiply(dt, mask, dt);
        }
        */

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

std::vector <cv::Mat> breath::gen_y_shifts(const cv::Mat frame_pre, const cv::Mat frame_cur, const cv::Rect roi) {
    if (frame_pre.empty() || frame_cur.empty()) {
        cv::Mat localshift = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdt = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);

        return std::vector < cv::Mat > {localshift, localdt};
    } else {
        cv::Mat frame_sub = frame_pre - frame_cur;
        cv::Mat frame_add = frame_pre + frame_cur;

        cv::Mat dy, dt, dydy, dydt;
        cv::Mat idydy, idydt, idt;

        cv::filter2D(frame_add, dy, CV_32FC1, param.kernel_dy, cv::Point(-1, -1), 0, cv::BORDER_REPLICATE);
        cv::divide(frame_sub, frame_add + 1e-3, dt);

        cv::multiply(dy, dy, dydy);
        cv::integral(dydy, idydy, CV_32FC1);
        cv::multiply(dy, dt, dydt);
        cv::integral(dydt, idydt, CV_32FC1);
        cv::integral(dt, idt, CV_32FC1);

        cv::Mat localshift = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);
        cv::Mat localdt = cv::Mat::zeros(param.bbox.size(), 1, CV_32FC1);

        float *idydy_data = (float *) idydy.data;
        float *idydt_data = (float *) idydt.data;

        for (int i = 0; i < param.bbox.size(); i++) {
            int x1 = param.bbox.at(i).x;
            int x2 = param.bbox.at(i).x + param.bbox.at(i).width - 1;
            int y1 = param.bbox.at(i).y;
            int y2 = param.bbox.at(i).y + param.bbox.at(i).height - 1;

            float sum_dydy = idydy_data[(y2 + 1) * idydy.cols + (x2 + 1)] + idydy_data[(y1) * idydy.cols + (x1)]
                             - idydy_data[(y2 + 1) * idydy.cols + (x1)] - idydy_data[(y1) * idydy.cols + (x2 + 1)];

            float sum_dydt = idydt_data[(y2 + 1) * idydt.cols + (x2 + 1)] + idydt_data[(y1) * idydt.cols + (x1)]
                             - idydt_data[(y2 + 1) * idydt.cols + (x1)] - idydt_data[(y1) * idydt.cols + (x2 + 1)];

            float sum_dt = idt.at<float>(y2 + 1, x2 + 1) + idt.at<float>(y1, x1) - idt.at<float>(y2 + 1, x1) -
                           idt.at<float>(y1, x2 + 1);

            float v = sum_dydt / (sum_dydy + 1e-6);

            localshift.at<float>(i, 0) = v;// / (1e-3 + sqrt(u * u + v * v));// ;
            localdt.at<float>(i, 0) = 100 * sum_dt / param.bbox.at(i).area();
        }

        return std::vector < cv::Mat > {localshift, localdt};
    }
}

/*
function S = combineXY(x, y)

A = [x; y];
R = A * A';
[u, s, ~] = svd(R);
S = u(:, 1) * u(:, 1)'*A;
S = mean(S, 1);

end
*/

cv::Mat combineXY(const cv::Mat trace_dx, const cv::Mat trace_dy) {
    cv::Mat A;
    cv::vconcat(trace_dx, trace_dy, A);

    cv::Mat R;
    cv::gemm(A, A, 1, cv::Mat(), 0, R, cv::GEMM_2_T);

    cv::Mat w, u, vt;
    cv::SVDecomp(R, w, u, vt, cv::SVD::FULL_UV); //cv::DECOMP_QR

    //std::cout << u.col(0) * u.col(0).t() << std::endl;

    cv::Mat S = u.col(0) * u.col(0).t() * A;
    cv::reduce(S, S, 0, cv::REDUCE_AVG);

    return S;
}

cv::Mat combineXY2(const cv::Mat trace_dx, const cv::Mat trace_dy) {
    static cv::Scalar dx_avg, dx_std, dy_avg, dy_std;
    cv::meanStdDev(trace_dx, dx_avg, dx_std);
    cv::meanStdDev(trace_dy, dy_avg, dy_std);  // 4-6ms

    static cv::Mat trace_dx_n, trace_dy_n;
    cv::subtract(trace_dx, (float) dx_avg[0], trace_dx_n);
    cv::subtract(trace_dy, (float) dy_avg[0], trace_dy_n);  // 6-9ms

    float sign = trace_dx_n.dot(trace_dy_n) > 0 ? 1 : -1;  // 约1ms
    float norm = sqrt(dx_std[0] * dx_std[0] + dy_std[0] * dy_std[0]);
    float xn = dx_std[0] / norm;
    float yn = (dy_std[0] * sign) / norm;
    float a = xn * xn + xn * yn;
    float b = yn * yn + xn * yn;

    cv::Mat S = a * trace_dx_n + b * trace_dy_n;  // 6-10 ms

    return S;
}


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
    cv::add(a_mat, b_mat, S);

    return S;
}

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


cv::Mat breath::gen_breath_trace(const cv::Mat dx, const cv::Mat dy, const cv::Rect chestroi) {
    update_buf(dx, localtrace_shift_dx, param.motion_len, true, false);
    update_buf(dy, localtrace_shift_dy, param.motion_len, true, false);

    // detrend and denoise
    if (localtrace_shift_dx.cols == param.motion_len) {
        // remove non-respiratory components
//		detrend(localtrace_shift_dx, param.detrend_mat);
//		detrend(localtrace_shift_dy, param.detrend_mat);

        /*
        * // Option 1: SVD based
        cv::Mat localtrace_shift;
        for (int i = 0; i < localtrace_shift_dx.rows; i++)
        {
            cv::Mat S = combineXY(localtrace_shift_dx.row(i), localtrace_shift_dy.row(i));
            localtrace_shift.push_back(S);
        }
        */

        // Option 2: STD based
//		cv::Mat localtrace_shift;
//        localtrace_shift.create(localtrace_shift_dx.rows, localtrace_shift_dx.cols, CV_32FC1);
//		for (int i = 0; i < localtrace_shift_dx.rows; i++)
//		{
//			cv::Mat S = combineXY2(localtrace_shift_dx.row(i), localtrace_shift_dy.row(i));
//			localtrace_shift.push_back(S);
//            S.copyTo(localtrace_shift.row(i));
//		}


        // Option 3: STD mat based
        cv::Mat localtrace_shift = combineXY3(localtrace_shift_dx, localtrace_shift_dy);


        // remove non-respiratory components
        detrend(localtrace_shift, param.detrend_mat);

        update_buf(localtrace_shift, localtrace_resp, param.localtrace_resp_len, false, true);

        /*
        // generate local resp
        update_buf(localtrace_shift_dx, localtrace_resp_dx, param.localtrace_resp_len, false);
        update_buf(localtrace_shift_dy, localtrace_resp_dy, param.localtrace_resp_len, false);

        localtrace_resp = cv::Mat();
        for (int i = 0; i < localtrace_resp_dx.rows; i++)
        {
            cv::Mat S = combineXY(localtrace_resp_dx.row(i), localtrace_resp_dy.row(i));
            localtrace_resp.push_back(S);
        }
        */
    }

    if (localtrace_resp.cols != param.localtrace_resp_len) {
        return cv::Mat();
    }

    // select by RoI
//	cv::Mat localtrace_sel;
//	if (chestroi.empty())
//	{
//		localtrace_sel = localtrace_resp.clone();
//	}
//	else
//	{
//		bbox_select.clear();
//		for (int i = 0; i < param.bbox.size(); i++)
//		{
//			float ratio_chest = ((float)(chestroi & param.bbox.at(i)).area() / (float)param.bbox.at(i).area());
//			if (ratio_chest > 0.8)
//			{
//				bbox_select.push_back(param.bbox.at(i));
//				localtrace_sel.push_back(localtrace_resp.row(i));
//			}
//		}
//	}
//
//
//	if (localtrace_sel.empty())
//	{
//		return cv::Mat();
//	}

//    // normalize local signals
//    cv::Mat localtrace_sel_n = localtrace_resp.clone();
//    for (int i = 0; i < localtrace_sel_n.rows; i++) {
//        cv::Scalar avg, stddev;
//        cv::meanStdDev(localtrace_sel_n.row(i), avg, stddev);
//        localtrace_sel_n.row(i) = (localtrace_sel_n.row(i) - avg[0]) / (1 + stddev[0]);
//    }

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

    /*
    cv::Mat im_plot = frame_raw.clone();
    for (int i = 0; i < this->bbox.size(); i++)
    {
        cv::rectangle(im_plot, this->bbox.at(i), cv::Scalar(255, 0, 0), 1);
    }
    cv::imshow("Sampling", im_plot);
    */
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

cv::Mat breath::plot_heatmap(const cv::Mat frame, cv::Mat softmask, const cv::Rect roi) {
    if (softmask.empty()) {
        return frame;
    } else {
        double st = (double) cv::getTickCount();
        cv::Mat heatmap = cv::Mat::zeros(frame.size(), CV_32FC1);
        cv::Mat accumap = cv::Mat::zeros(frame.size(), CV_32FC1);
        for (int i = 0; i < param.bbox.size(); i++) {
            cv::add(heatmap(param.bbox.at(i)), cv::Scalar(softmask.at<float>(i, 0)), heatmap(param.bbox.at(i)));
            cv::add(accumap(param.bbox.at(i)), cv::Scalar(1), accumap(param.bbox.at(i)));
        }
        double en = (double) cv::getTickCount();
        double T = (en - st) / cv::getTickFrequency();
        std::cout << "TTTTTTTTTTT = " << T << std::endl;

        cv::divide(heatmap, accumap + 1e-6, heatmap);

        cv::normalize(heatmap, heatmap, 0, 255, cv::NORM_MINMAX, CV_32FC1);
        heatmap.convertTo(heatmap, CV_8UC1);
        cv::applyColorMap(heatmap, heatmap, cv::COLORMAP_JET);

        cv::Mat frame_heatmap;
        if (frame.channels() == 1) {
            cv::cvtColor(frame, frame_heatmap, cv::COLOR_GRAY2RGB);
        } else {
            frame.copyTo(frame_heatmap);
        }

        cv::addWeighted(heatmap, 0.75, frame_heatmap, 0.25, 0, frame_heatmap);

        /*
        for (int i = 0; i < this->bbox.size(); i++)
        {
            float ratio = ((float)(roi & bbox.at(i)).area() / (float)bbox.at(i).area());


            if (ratio > faceroi_thr)
            {
                cv::rectangle(frame_heatmap, bbox.at(i), cv::Scalar(255, 255, 255), 1);
            }
            //else
            //cv::rectangle(frame_heatmap, bbox.at(i), cv::Scalar(0, 0, 255), 1);
        }
        */

        return frame_heatmap;
    }
}

