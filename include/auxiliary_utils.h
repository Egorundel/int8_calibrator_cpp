#pragma once
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "NvInfer.h"

std::vector<int> getTensorSizes(nvinfer1::INetworkDefinition* network);