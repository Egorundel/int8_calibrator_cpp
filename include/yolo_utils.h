#pragma once
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "NvInfer.h"

void LetterBox(const cv::Mat& image, cv::Mat& outImage,
	cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
	const cv::Size& newShape,
	bool autoShape = false,
	bool scaleFill = false,
	bool scaleUp = true,
	int stride = 32,
	const cv::Scalar& color = cv::Scalar(0, 0, 0)
);

std::vector<int> getTensorSizes(nvinfer1::INetworkDefinition* network);
