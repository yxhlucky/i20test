#include "bin2mat.h"

bin2mat::bin2mat(const std::string path)
{
	fs = std::ifstream(path, std::fstream::binary);
	if (!fs.is_open())
	{
		std::cout << "WARNING bin2mat, error opening " << path << ", no action done\n";
		return;
	}
	// get length of file:
	fs.seekg(0, std::ios::end);
	size_t length = fs.tellg();
	fs.seekg(0, std::ios::beg);

	size_t headerLength = 4 * sizeof(int);

	if (length < headerLength)
	{
		std::cout << "WARNING bin2mat, missing or corrupt header in " << path << ", no action done\n";
		return;
	}

	// Header
	fs.read((char*)&rows, sizeof(int));
	fs.read((char*)&cols, sizeof(int));
	fs.read((char*)&channels, sizeof(int));
	fs.read((char*)&depth, sizeof(int));

	std::cout << depth << std::endl;

	bytesPerRow = CV_ELEM_SIZE1(depth) * cols * channels;
	rowCount = (length - headerLength) / bytesPerRow;
	frameCount = rowCount / rows;

	std::cout << "rows = " << rows << ", cols = " << cols << ", depth = " << depth << ", channels = " << channels << std::endl;
	std::cout << "type = " << CV_MAKETYPE(depth, channels) << ", real type = " << CV_16UC3 << std::endl;

}

void bin2mat::set_frame(int frame_number)
{
	if (frame_number < 0 || frame_number >= frameCount)
	{
		std::cout << "Invalid frame number. Frame number should be between 0 and " << frameCount - 1 << std::endl;
		return;
	}

	size_t headerLength = 4 * sizeof(int);
	size_t frame_data_offset = headerLength + frame_number * rows * bytesPerRow;

	fs.seekg(frame_data_offset, std::ios::beg);
}


cv::Mat bin2mat::get_mat()
{
	cv::Mat mat = cv::Mat(rows, cols, CV_MAKETYPE(depth, channels));
	fs.read((char*)mat.data, rows * bytesPerRow);

	return mat;
}

void bin2mat::close()
{
	fs.close();
}
