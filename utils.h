//
// Created by rfv56 on 2023/11/30.
//

#ifndef __LOADCSV_H
#define __LOADCSV_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>
#include "debayer.h"
#include "bin2mat.h"
#include "pulse.h"

cv::Mat drawSignalPlot(const std::vector<float>& signal, int bpm, int width, int height, int maxPoints);

void update_signal_trace(float val, cv::Mat &trace, int len);

void saveROIs(const std::vector<cv::Rect>& rois, const std::string& filename);

void loadROIs(std::vector<cv::Rect>& rois, const std::string& filename);

cv::Rect resizeRect(cv::Rect src);

void processImage(cv::Mat& im, cv::Mat& bgr, debayer& db, bin2mat& b2m);


#endif
