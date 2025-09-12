#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

class bin2mat
{
public:
	bin2mat(const std::string path);
	void set_frame(int frame_number);
	cv::Mat get_mat();
	void close();

	size_t frameCount;
	int rows, cols, channels, depth;

private:
	std::ifstream fs;
	size_t bytesPerRow, rowCount;
};
