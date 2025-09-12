#include "pulse.h"

pulse::pulse(const cv::Size frame_size, const int fps) {
    param.fps = fps;
    param.frame_size = frame_size;
    param.win_len = fps * 3; // 3 sec//4sec
    param.rate_len = fps * 10; // 10 sec;
    param.live_len = fps * 5;
    param.heartrate_buffer_len = fps * 10;
    gen_bbox(frame_size.width, frame_size.height, bbox);
//    hann = create_hann(param.rate_len);
}

pulse_info pulse::detect(const cv::Mat rgb, const cv::Rect faceroi, const bool night_flag) {
    pulse_info pi;

    // generate spatial mean
    cv::Mat rgb_triplet = gen_color_triplet(rgb, bbox);

    // generate temporal traces
    concatenate(rgb_triplet, rgb_trace, param.win_len);

    // process
    if (rgb_trace.cols == param.win_len) {
//        cv::Mat rgb_trace_sel;
        cv::Mat ppg_bgr;

        if (bbox.size() > 0) {
            cv::Mat trace = extract_ppg(rgb_trace, night_flag);
            cv::Mat ppg_trace_bgr = trace;

            cv::Mat snr_bgr = gen_snr(ppg_trace_bgr);
            cv::Mat sort_idx_bgr;
            cv::sortIdx(snr_bgr, sort_idx_bgr, cv::SORT_DESCENDING + cv::SORT_EVERY_COLUMN);

            bbox_topK_bgr.clear();
            /********************Liveness Detection********************************/
            std::vector<int> valid_indices; // 用于存储满足条件的索引
            for (int i = 0; i < snr_bgr.rows; ++i) {
                // 判断每行的第一个元素是否大于信噪比阈值
                if (snr_bgr.at<float>(i, 0) > param.snr_threshold) {
                    valid_indices.push_back(i); // 如果满足条件，保存当前索引
                }
            }

            int num_valid_boxes = valid_indices.size(); // 获取满足条件的索引个数
            // 计算有效框的比例（有效框数 / 总框数）
            float ratio = static_cast<float>(num_valid_boxes) / bbox.size();

            ratio_trace.push_back(ratio); // 将当前的有效比例添加到跟踪列表
            // 如果跟踪列表长度超过设定值，则移除最早的一个比例
            if (ratio_trace.size() > param.live_len) {
                ratio_trace.erase(ratio_trace.begin());
            }

            // 计算跟踪列表中的比例的平均值作为平滑后的比例
            smoothed_ratio = std::accumulate(ratio_trace.begin(), ratio_trace.end(), 0.0f) / ratio_trace.size();

            /**************************************************************************/
            if (num_valid_boxes < sort_idx_bgr.rows) {
                cv::Mat ppg_sel_bgr;
                //int min_box = std::min(param.select_topK, num_valid_boxes);
                if (num_valid_boxes < 10) {
                    for (int i = 0; i < param.select_topK; i++) {
                        ppg_sel_bgr.push_back(ppg_trace_bgr.row(sort_idx_bgr.at<int>(i, 0)));
                        bbox_topK_bgr.push_back(bbox.at(sort_idx_bgr.at<int>(i, 0))); // 将对应的边界框添加到 box_sel

                    }
                } else {
                    for (int i = 0; i < num_valid_boxes; ++i) {
                        ppg_sel_bgr.push_back(ppg_trace_bgr.row(valid_indices[i]));

                        bbox_topK_bgr.push_back(bbox.at(valid_indices[i]));
                    }
                }
                cv::reduce(ppg_sel_bgr, ppg_bgr, 0, cv::REDUCE_AVG);
            } else {
                cv::reduce(ppg_trace_bgr, ppg_bgr, 0, cv::REDUCE_AVG);
            }
            // normalize
            cv::Scalar avg, stddev;
            cv::meanStdDev(ppg_bgr, avg, stddev);
            ppg_bgr = (ppg_bgr - avg[0]) / (stddev[0]);
        } else {
            ppg_bgr = cv::Mat::zeros(1, rgb_trace.cols, CV_32FC1);
        }

        // overlap add
        overlap_add(ppg_bgr, PPG_bgr, param.rate_len);

        // calculate HR and Q
        calc_heartrate(PPG_bgr, pi.heartrate);
        pi.ppg_signal = (std::vector < float > )(PPG_bgr.reshape(1, 1));
    }

    return pi;
}

void pulse::concatenate(const cv::Mat val, cv::Mat &trace, int len) {
    if (val.empty()) return; // 检查输入是否为空

    if (trace.empty()) {
        val.copyTo(trace); // 初始化trace
    } else {
        if (trace.cols < len) {
            cv::hconcat(trace, val, trace); // 拼接新的列
        } else {
            // 如果trace已经有足够的列数，使用预分配的临时矩阵来避免频繁分配内存
            static cv::Mat temp(trace.rows, len, trace.type()); // 静态变量，预分配内存
            trace.colRange(1, trace.cols).copyTo(temp.colRange(0, len - 1)); // 左移数据
            val.copyTo(temp.col(len - 1)); // 添加新列
            temp.copyTo(trace); // 更新trace
        }
    }
}

cv::Mat pulse::gen_snr(const cv::Mat trace)
{
    cv::Mat fft, fft_mag;
    cv::Mat tmp[2] = { trace, cv::Mat::zeros(trace.size(), trace.type()) };
    cv::merge(tmp, 2, fft);
    cv::dft(fft, fft, cv::DFT_SCALE + cv::DFT_COMPLEX_OUTPUT + cv::DFT_ROWS);

    cv::split(fft.colRange(0, floor(fft.cols / 2) - 1), tmp);
    cv::magnitude(tmp[0], tmp[1], fft_mag);

    cv::Mat spec_sum, spec_max, spec_res;
    cv::reduce(fft_mag, spec_sum, 1, cv::REDUCE_SUM);
    cv::reduce(fft_mag, spec_max, 1, cv::REDUCE_MAX);

    cv::subtract(spec_sum, spec_max, spec_res);

    cv::Mat snr;
    cv::divide(spec_max, spec_res + 1e-3, snr);

    return snr;
}

cv::Mat pulse::create_hann(int win_len) {
    cv::Mat hann = cv::Mat(1, win_len, CV_32FC1);

    for (int i = 0; i < win_len; i++) {
        hann.at<float>(0, i) = 0.5 * (1 - cos((float) 2 * 3.14159265358979323846f * i / ((float) win_len - 1)));
    }

    return hann;
}

void pulse::gen_bbox(const int frame_width, const int frame_height, std::vector <cv::Rect> &bbox) {
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
}

cv::Mat pulse::gen_color_triplet(const cv::Mat rgb, const std::vector <cv::Rect> bbox) {
    if (rgb.empty()) {
        return cv::Mat::zeros(bbox.size(), 1, CV_32FC3);//？这里接收一个空图什么情况会发生？
    } else {
        cv::Mat rgb_f;
        rgb.convertTo(rgb_f, CV_32FC3);

        cv::Mat rgb_int;
        cv::integral(rgb_f, rgb_int, CV_32FC3);

        cv::Mat rgb_triplet = cv::Mat::zeros(bbox.size(), 1, CV_32FC3);

        for (int i = 0; i < bbox.size(); i++) {
            int x1 = bbox.at(i).x;
            int x2 = bbox.at(i).x + bbox.at(i).width - 1;
            int y1 = bbox.at(i).y;
            int y2 = bbox.at(i).y + bbox.at(i).height - 1;

            rgb_triplet.at<cv::Vec3f>(i, 0) = get_bbox_sum(rgb_int, x1, y1, x2, y2);
        }

        return rgb_triplet;
    }
}

cv::Vec3f pulse::get_bbox_sum(const cv::Mat mat, int x1, int y1, int x2, int y2) {
    // Ensure the indices are within bounds
    x1 = std::max(x1, 0);
    y1 = std::max(y1, 0);
    x2 = std::min(x2, mat.cols - 1);
    y2 = std::min(y2, mat.rows - 1);

    cv::Vec3f vec_4 = mat.at<cv::Vec3f>(y2 + 1, x2 + 1);
    cv::Vec3f vec_3 = mat.at<cv::Vec3f>(y2 + 1, x1);
    cv::Vec3f vec_2 = mat.at<cv::Vec3f>(y1, x2 + 1);
    cv::Vec3f vec_1 = mat.at<cv::Vec3f>(y1, x1);
    cv::Vec3f vec = vec_4 + vec_1 - vec_2 - vec_3;

    return vec;
}


void normalize_dc(cv::Mat trace) {
    cv::Mat trace_avg, trace_avg_mat;
    cv::reduce(trace, trace_avg, 1, cv::REDUCE_AVG);
    cv::repeat(trace_avg, 1, trace.cols, trace_avg_mat);
    cv::divide(trace, trace_avg_mat, trace);
}

cv::Mat pulse::extract_ppg(cv::Mat& rgb_trace, const bool night_flag)
{
    // Extract and preprocess color channels
    std::vector<cv::Mat> color_channels(3);

    for (int i = 0; i < 3; i++)
    {
        cv::extractChannel(rgb_trace, color_channels[i], i);
        normalize_dc(color_channels[i]); // Normalize DC component
        bandpass(color_channels[i], param.bandpassfilt); // Apply bandpass filter
    }

    // Initialize matrices for processing
    cv::Mat trace = cv::Mat::zeros(3, rgb_trace.cols, CV_32FC1);
    cv::Mat ppg_trace;

    // Process each row
    for (int i = 0; i < rgb_trace.rows; i++)
    {
        color_channels[0].row(i).copyTo(trace.row(0));
        color_channels[1].row(i).copyTo(trace.row(1));
        color_channels[2].row(i).copyTo(trace.row(2));
        cv::Mat ppg;

        if (night_flag)
        {

            // MEAN method
            cv::reduce(trace, ppg, 0, cv::REDUCE_AVG);
        }
        else
        {

            // PBV method
//            double bias = 1e-8; // this parameter can be tuned
//            cv::Mat A, B, C;
//            cv::gemm(trace, trace, 1, cv::Mat(), 0, A, cv::GEMM_2_T);
//            B = cv::Mat::eye(cv::Size(3, 3), CV_32FC1);
//            cv::scaleAdd(B, bias, A, C);
//            cv::gemm(param.pbv.colRange(0, 3), C.inv(cv::DecompTypes::DECOMP_SVD), 1, cv::Mat(), 0, C);
//            cv::gemm(C, trace, 1, cv::Mat(), 0, ppg);

            ppg = trace.row(1);
        }
        // Normalize PPG signal
        cv::Scalar avg, stddev;
        cv::meanStdDev(ppg, avg, stddev);
        ppg = (ppg - avg[0]) / (1e-3 + stddev[0]);

        ppg_trace.push_back(ppg);
    }

    return  ppg_trace;
}
void pulse::overlap_add(const cv::Mat val, cv::Mat &buf, int len) {
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

void pulse::overlap_add(const std::vector<float> val, std::vector<float> &buf, int len) {
    if (val.empty()) return;

    if (buf.empty()) {
        buf = val;
    } else {
        buf.push_back(0);

        for (int i = 0; i < val.size(); i++) {
            buf.at(buf.size() - val.size() + i) += val.at(i);
        }

        if (buf.size() > len) {
            buf.erase(buf.begin());
        }
    }
}

//void pulse::calc_heartrate(const cv::Mat pulse, int &heartrate) {
//    if (pulse.empty()) {
//        heartrate = 0;
//
//        return;
//    }
//
//    // zero mean
//    cv::Mat pulse_n;
//    cv::subtract(pulse, cv::mean(pulse), pulse_n);
//
//    // zero padding
//    cv::Mat pulse_zp;
//    cv::hconcat(pulse_n, cv::Mat::zeros(pulse_n.rows, param.fps * 60 - pulse_n.cols, CV_32FC1), pulse_zp);
//
//    // spectrum
//    cv::Mat spec = gen_spectrum(pulse_zp);
//
//    // pulse rate
//    int rate = -1;
//    double maxval = 0;
//    cv::minMaxIdx(spec.t(), 0, &maxval, 0, &rate);
//    heartrate = rate;
//
//
//    if (heartrate >= 0 && heartrate <= 40) // i20
//        std::cout<<"coming!!!!"<<std::endl;
//        if (heartrate >= param.min_heartrate && heartrate <= param.max_heartrate) {
//            int bins = 50;
//            cv::Mat hist;
//            int histSize[] = {bins};
//            float hrRange[] = {float(param.min_heartrate), float(param.max_heartrate)};
//            const float* ranges[] = {hrRange};
//            cv::calcHist(&heartrate_trace, 1, 0, cv::Mat(), hist, 1, histSize, ranges);
//
//            int hist_w = 1800;
//            int hist_h = 500;
//            int bin_w = cvRound((double) hist_w / bins);
//            cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0));
//
//            double max_val = 0;
//            cv::minMaxLoc(hist, 0, &max_val);
//
//            int maxBinIndex = 0;
//            for (int i = 1; i < bins; i++) {
//                if (hist.at<float>(i) > hist.at<float>(maxBinIndex)) {
//                    maxBinIndex = i;
//                }
//                cv::line(histImage,
//                         cv::Point(bin_w * (i - 1), hist_h - cvRound(hist.at<float>(i - 1) * hist_h / max_val)),
//                         cv::Point(bin_w * i, hist_h - cvRound(hist.at<float>(i) * hist_h / max_val)),
//                         cv::Scalar(255), 2, 8, 0);
//            }
//
//
//            int maxBinRangeStart = cvRound(hrRange[0] + maxBinIndex * (hrRange[1] - hrRange[0]) / bins);
//            int maxBinRangeEnd = cvRound(hrRange[0] + (maxBinIndex + 1) * (hrRange[1] - hrRange[0]) / bins);
//            std::cout << "The range of the most frequent bin is: " << maxBinRangeStart << " to " << maxBinRangeEnd << std::endl;
//
//            if (heartrate_trace.cols >= param.heartrate_buffer_len) {
//                if (heartrate >= maxBinRangeStart -5 && heartrate <= maxBinRangeEnd+5) {
//                    update_rate_trace(cv::Mat(1, 1, CV_32FC1, (float)heartrate), heartrate_trace, param.heartrate_buffer_len);
//                }else{
//                    std::cout << "1111111 "  << std::endl;
//
//                }
//            } else {
//                update_rate_trace(cv::Mat(1, 1, CV_32FC1, (float)heartrate), heartrate_trace, param.heartrate_buffer_len);
//            }
//            //update_rate_trace(cv::Mat(1, 1, CV_32FC1, (float)heartrate), heartrate_trace, param.heartrate_buffer_len);
//
//            for (int i = 0; i < bins; i++) {
//                int bin_value = cvRound(hrRange[0] + i * (hrRange[1] - hrRange[0]) / bins);
//                int bin_count = cvRound(hist.at<float>(i));
//
//                std::string value_text = std::to_string(bin_value);
//                int value_text_x = bin_w * i + 5;
//                int value_text_y = hist_h - 20;
//                cv::putText(histImage, value_text,
//                            cv::Point(value_text_x, value_text_y),
//                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 1, 8, false);
//
//                std::string count_text = std::to_string(bin_count);
//                int count_text_y = hist_h - 5;
//                cv::putText(histImage, count_text,
//                            cv::Point(value_text_x, count_text_y),
//                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255), 1, 8, false);
//            }
//
//            cv::imshow("Heart Rate Histogram", histImage);
//            cv::waitKey(1);
//        }
//
//
//    if (heartrate_trace.empty()) {
//        heartrate = 0;
//    } else {  // max_hr - min_hr < 10
//        heartrate =  (int) cv::mean(heartrate_trace)[0];
//    }
//
//
//
//}





void pulse::calc_heartrate(const cv::Mat pulse, int &heartrate) {
    if (pulse.empty()) {
        heartrate = 0;
        return;
    }

    // zero mean
    cv::Mat pulse_n;
    cv::subtract(pulse, cv::mean(pulse), pulse_n);

    // zero padding
    cv::Mat pulse_zp;
    cv::hconcat(pulse_n, cv::Mat::zeros(pulse_n.rows, param.fps * 60 - pulse_n.cols, CV_32FC1), pulse_zp);

    // spectrum
    cv::Mat spec = gen_spectrum(pulse_zp);

    // pulse rate
    int rate = -1;
    double maxval = 0;
    cv::minMaxIdx(spec.t(), 0, &maxval, 0, &rate);
    heartrate = rate;
    int max_count = 0;
    int max_bin_index = -1;
    int avg_in_max_bin = 0;
    int bins_above_10_count = 0;
    std::vector<float> values_in_max_bin;
    std::unordered_map<int, int> peak_counts;
    float stdev = 0;
    if (heartrate >= param.min_heartrate && heartrate <= param.max_heartrate) {
        int bins = 20;
        cv::Mat hist;
        int histSize[] = {bins};
        float hrRange[] = {float(param.min_heartrate), float(param.max_heartrate)};
        const float *ranges[] = {hrRange};
        cv::calcHist(&heartrate_trace, 1, 0, cv::Mat(), hist, 1, histSize, ranges);

        double max_val = 0;
        cv::minMaxLoc(hist, 0, &max_val);

        int maxBinIndex = 0;
        for (int i = 1; i < bins; i++) {
            if (hist.at<float>(i) > hist.at<float>(maxBinIndex)) {
                maxBinIndex = i;
            }
        }

        int maxBinRangeStart = cvRound(hrRange[0] + maxBinIndex * (hrRange[1] - hrRange[0]) / bins);
        int maxBinRangeEnd = cvRound(hrRange[0] + (maxBinIndex + 1) * (hrRange[1] - hrRange[0]) / bins);
        std::vector<float> values_in_max_bin;

        for (int i = 0; i < heartrate_trace.cols; ++i) {
            float value = heartrate_trace.at<float>(0, i);
            if (value >= maxBinRangeStart && value <= maxBinRangeEnd) {
                values_in_max_bin.push_back(value);
            }
        }

        if (!values_in_max_bin.empty()) {
            avg_in_max_bin = std::accumulate(values_in_max_bin.begin(), values_in_max_bin.end(), 0.0f) /
                             values_in_max_bin.size();
        }

        update_rate_trace(cv::Mat(1, 1, CV_32FC1, (float) heartrate), heartrate_trace, param.heartrate_buffer_len);

        for (int i = 0; i < bins; i++) {

            int bin_value = cvRound(hrRange[0] + i * (hrRange[1] - hrRange[0]) / bins);
            int bin_count = cvRound(hist.at<float>(i));

            if (bin_count > 10) {  // 当前簇的计数
                peak_counts[bin_value] += bin_count;
                bins_above_10_count++;
            }
        }
        stdev = calculateWeightedStd(peak_counts);
    }

    if (heartrate_trace.empty() || stdev > 8) {
        heartrate = 0;
    } else {  // max_hr - min_hr < 10
        heartrate = avg_in_max_bin;
    }

}

void pulse::update_rate_trace(const cv::Mat val, cv::Mat &trace, int len) {
    if (val.empty()) return;

    if (trace.empty()) {
        val.copyTo(trace);
    } else {
        if (trace.cols < len) {
            cv::hconcat(trace, val.col(val.cols - 1), trace);
        } else {
            cv::hconcat(trace.colRange(1, trace.cols), val.col(val.cols - 1), trace);
        }
    }
}

cv::Mat pulse::gen_spectrum(const cv::Mat trace)
{
    cv::Mat fft;
    cv::Mat tmp[2] = { trace, cv::Mat::zeros(trace.size(), trace.type()) };
    cv::merge(tmp, 2, fft);
    cv::dft(fft, fft, cv::DFT_SCALE + cv::DFT_COMPLEX_OUTPUT + cv::DFT_ROWS);

    cv::Mat fft_mag;
    cv::split(fft.colRange(0, floor(fft.cols / 2) - 1), tmp);
    cv::magnitude(tmp[0], tmp[1], fft_mag);

    // release
    tmp->release();

    return fft_mag;
}


void pulse::bandpass(cv::Mat& trace, const float band[2])
{
    cv::Mat fft;
    cv::Mat tmp[2] = { trace, cv::Mat::zeros(trace.size(), trace.type()) };
    cv::merge(tmp, 2, fft);
    cv::dft(fft, fft, cv::DFT_SCALE + cv::DFT_COMPLEX_OUTPUT + cv::DFT_ROWS);

    cv::Mat fft_band = cv::Mat::zeros(fft.size(), CV_32FC2);
    fft.colRange(band[0], band[1]).copyTo(fft_band.colRange(band[0], band[1]));
    cv::dft(fft_band, trace, cv::DFT_SCALE + cv::DFT_REAL_OUTPUT + cv::DFT_ROWS + cv::DFT_INVERSE);

    // release
    tmp->release();
}



