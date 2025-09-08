#include "utils.h"


cv::Mat drawSignalPlot(const std::vector<float>& signal, int bpm, int width , int height, int maxPoints) {
    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    if (signal.empty()) return plot;
    // 确定要显示的点数量
    int numPoints = std::min((int)signal.size(), maxPoints);
    int startIdx = signal.size() > maxPoints ? signal.size() - maxPoints : 0;
    // 找出信号的最大最小值以进行缩放
    double minVal = *std::min_element(signal.begin() + startIdx, signal.end());
    double maxVal = *std::max_element(signal.begin() + startIdx, signal.end());
    double range = maxVal - minVal;
    if (range < 0.0001) range = 1.0; // 防止除以零
    // 绘制坐标轴
    cv::line(plot, cv::Point(0, height/2), cv::Point(width, height/2), cv::Scalar(50, 50, 50), 1);
    // 绘制信号
    for (int i = 0; i < numPoints - 1; i++) {
        double val1 = signal[startIdx + i];
        double val2 = signal[startIdx + i + 1];
        // 缩放到绘图区域
        int x1 = i * width / numPoints;
        int x2 = (i + 1) * width / numPoints;
        int y1 = height - (int)((val1 - minVal) / range * (height * 0.8) + height * 0.1);
        int y2 = height - (int)((val2 - minVal) / range * (height * 0.8) + height * 0.1);
        cv::line(plot, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
    }
    // 添加当前值
    if (!signal.empty()) {
        std::string valueText = "Current: " + std::to_string(bpm) + " bpm";
        cv::putText(plot, valueText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 255, 255), 2);
    }
    return plot;
}

void update_signal_trace(float val, cv::Mat &trace, int len) {
    if (trace.empty()) {
        trace = cv::Mat(1, 1, CV_32F, val);
    } else {
        cv::Mat new_val(1, 1, CV_32F, val);
        if (trace.rows < len) {
            cv::vconcat(trace, new_val, trace);  // 垂直拼接
        } else {
            cv::vconcat(trace.rowRange(1, trace.rows), new_val, trace);  // 移除最旧的行，添加新行
        }
    }
}


void saveROIs(const std::vector<cv::Rect>& rois, const std::string& filename)
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file to save ROIs";
        return;
    }

    for (const cv::Rect& roi : rois) {
        file << roi.x << "," << roi.y << "," << roi.width << "," << roi.height << "\n";
    }

    file.close();
}

void loadROIs(std::vector<cv::Rect>& rois, const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Unable to open file to load ROIs";
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        cv::Rect roi;
        std::getline(ss, item, ','); roi.x = std::stoi(item);
        std::getline(ss, item, ','); roi.y = std::stoi(item);
        std::getline(ss, item, ','); roi.width = std::stoi(item);
        std::getline(ss, item, ','); roi.height = std::stoi(item);
        rois.push_back(roi);
    }

    file.close();
}

cv::Rect resizeRect(cv::Rect src)
{
    return cv::Rect(src.x, src.y, src.width, src.height);
}

void processImage(cv::Mat& im, cv::Mat& bgr, debayer& db, bin2mat& b2m)
{
    if (b2m.channels == 1)
    {
        db.raw2bgr(im, bgr);

        //cv::resize(bgr, bgr, cv::Size(240, 135), 0, 0, cv::INTER_LINEAR);
    }
    else{

        bgr = im;
    }
}

