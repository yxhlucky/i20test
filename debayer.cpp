#include "debayer.h"

debayer::debayer(raw_format format)
{
    this->format = format;
}

void debayer::raw2bgr(cv::Mat &src, cv::Mat &dst)
{
    int rows = src.rows;
    int cols = src.cols;

    dst = cv::Mat::zeros(rows / 2, cols / 2, CV_8UC3); // ����һ���µ�3ͨ������

    for (int y = 0; y < (rows >> 1); y++)
    {
        uchar* S = src.data + ((cols * y) << 1);
        uchar* nextS = S + cols;
        for (int x = 0; x < (cols >> 1); x++)
        {
            int index = x << 1;

            // �������ݵ� dst ����
            if (format == raw_format::RGGB)
            {
                dst.at<cv::Vec3b>(y, x)[0] = nextS[index + 1]; // B
                dst.at<cv::Vec3b>(y, x)[1] = (S[index + 1] + nextS[index]) / 2; // Take average of G
                //dst.at<cv::Vec3b>(y, x)[1] = S[index + 1]; // Take average of G
                dst.at<cv::Vec3b>(y, x)[2] = S[index]; // R
            }
            else if (format == raw_format::BGGR)
            {
                dst.at<cv::Vec3b>(y, x)[0] = S[index]; // B
                dst.at<cv::Vec3b>(y, x)[1] = (S[index + 1] + nextS[index]) / 2; // Take average of G
                dst.at<cv::Vec3b>(y, x)[2] = nextS[index + 1]; // R
            }
            else if (format == raw_format::GRBG)
            {
                dst.at<cv::Vec3b>(y, x)[0] = nextS[index]; // B
                dst.at<cv::Vec3b>(y, x)[1] = (S[index] + nextS[index + 1]) / 2; // Take average of G
                //dst.at<cv::Vec3b>(y, x)[1] = S[index]; // Take average of G
                dst.at<cv::Vec3b>(y, x)[2] = S[index + 1]; // R
            }
            else if (format == raw_format::GBRG)
            {
                dst.at<cv::Vec3b>(y, x)[0] = S[index + 1]; // B
                dst.at<cv::Vec3b>(y, x)[1] = (S[index] + nextS[index + 1]) / 2; // Take average of G
                //dst.at<cv::Vec3b>(y, x)[1] = S[index]; // Take average of G
                dst.at<cv::Vec3b>(y, x)[2] = nextS[index]; // R
            }
        }
    }
}
void debayer::bgr2raw(const cv::Mat &bgr, cv::Mat &raw, raw_format format){
    int rows = bgr.rows;
    int cols = bgr.cols;

    // ����һ���µĵ�ͨ������
    raw = cv::Mat::zeros(rows * 2, cols * 2, CV_8UC1);

    for (int y = 0; y < rows; y++) {
        const uchar* S = bgr.ptr<uchar>(y);
        uchar* D = raw.ptr<uchar>(2 * y);
        uchar* nextD = raw.ptr<uchar>(2 * y + 1);

        for (int x = 0; x < cols; x++) {
            int index = x * 2;

            // ���ݲ�ͬ�ĸ�ʽ���в�ͬ��ͨ������
            if (format == raw_format::GBRG) {
                D[index] = S[3 * x + 1];        // G
                D[index + 1] = S[3 * x + 0];    // B
                nextD[index] = S[3 * x + 2];    // R
                nextD[index + 1] = S[3 * x + 1];// G
            } else if (format == raw_format::RGGB) {
                D[index] = S[3 * x + 2];        // R
                D[index + 1] = S[3 * x + 1];    // G
                nextD[index] = S[3 * x + 1];    // G
                nextD[index + 1] = S[3 * x + 0];// B
            } else if (format == raw_format::BGGR) {
                D[index] = S[3 * x + 0];        // B
                D[index + 1] = S[3 * x + 1];    // G
                nextD[index] = S[3 * x + 1];    // G
                nextD[index + 1] = S[3 * x + 2];// R
            } else if (format == raw_format::GRBG) {
                D[index] = S[3 * x + 1];        // G
                D[index + 1] = S[3 * x + 2];    // R
                nextD[index] = S[3 * x + 0];    // B
                nextD[index + 1] = S[3 * x + 1];// G
            }
        }
    }
}
void debayer::raw2bgra(cv::Mat &src, cv::Mat &dst)
{
    int rows = src.rows;
    int cols = src.cols;

    dst = cv::Mat::zeros(rows / 2, cols / 2, CV_8UC4); // ����һ���µ�4ͨ������

    for (int y = 0, i = 0; y < (rows >> 1); y++)
    {
        uchar* S = src.data + ((cols * y) << 1);
        uchar* nextS = S + cols;
        for (int x = 0; x < (cols >> 1); x++)
        {
            int index = x << 1;

            // �������ݵ� dst ����
            if (format == raw_format::BAGR)
            {
                dst.at<cv::Vec4b>(y, x)[0] = S[index]; // B
                dst.at<cv::Vec4b>(y, x)[1] = nextS[index]; // G
                dst.at<cv::Vec4b>(y, x)[2] = nextS[index + 1]; // R
                dst.at<cv::Vec4b>(y, x)[3] = S[index + 1]; // A
            }
        }
    }
}
