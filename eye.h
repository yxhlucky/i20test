#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric>

struct sleep_info
{
	//std::string state; // sleep | awake
	int state; // sleep is 0 | awake is 1
	float score;
};

struct sleep_param
{
	int actbuf_len;
	int eyebuf_len;
	std::string classifier_name;
};

class eye
{
public:
	eye(std::string classifier_name, std::string model_path, int fps);
	sleep_info detect(cv::Mat& rgb, cv::Rect faceroi, cv::Point2i lefteye, cv::Point2i righteye, float activity);

private:
	sleep_param param;
	cv::HOGDescriptor hog;
	cv::Ptr<cv::ml::SVM> svm;
	cv::Ptr<cv::ml::ANN_MLP> mlp;
	std::deque<float> eyebuf;
	std::deque<float> actbuf;

	cv::Mat gen_hogfeat(cv::Mat rgb, cv::Mat faces);
	cv::Mat extract_hogfeat(cv::Mat& rgb, cv::Rect faceroi, cv::Point2i lefteye, cv::Point2i righteye);
	cv::Mat extract_hogfeat2(cv::Mat& rgb, cv::Rect faceroi, cv::Point2i lefteye, cv::Point2i righteye);

	float classify_hogfeat(cv::Mat feat);
};
