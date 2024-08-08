#include "yolo_utils.h"
using namespace cv;
using namespace std;

void LetterBox(const cv::Mat& image, cv::Mat& outImage, cv::Vec4d& params, const cv::Size& newShape,
	bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color)
{
	if (false) {
		int maxLen = MAX(image.rows, image.cols);
		outImage = Mat::zeros(Size(maxLen, maxLen), CV_8UC3);
		image.copyTo(outImage(Rect(0, 0, image.cols, image.rows)));
		params[0] = 1;
		params[1] = 1;
		params[3] = 0;
		params[2] = 0;
	}

	cv::Size shape = image.size();
	float r = std::min((float)newShape.height / (float)shape.height,
		(float)newShape.width / (float)shape.width);
	if (!scaleUp)
		r = std::min(r, 1.0f);

	float ratio[2]{ r, r };
	int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

	auto dw = (float)(newShape.width - new_un_pad[0]);
	auto dh = (float)(newShape.height - new_un_pad[1]);

	if (autoShape)
	{
		dw = (float)((int)dw % stride);
		dh = (float)((int)dh % stride);
	}
	else if (scaleFill)
	{
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = newShape.width;
		new_un_pad[1] = newShape.height;
		ratio[0] = (float)newShape.width / (float)shape.width;
		ratio[1] = (float)newShape.height / (float)shape.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
	{
		cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
	}
	else {
		outImage = image.clone();
	}

	int top = int(std::round(dh - 0.1f));
	int bottom = int(std::round(dh + 0.1f));
	int left = int(std::round(dw - 0.1f));
	int right = int(std::round(dw + 0.1f));
	params[0] = ratio[0];
	params[1] = ratio[1];
	params[2] = left;
	params[3] = top;
	cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

std::vector<int> getTensorSizes(nvinfer1::INetworkDefinition* network) {
    // Vector to store tensor names and dimensions
    std::vector<std::pair<std::string, std::vector<int>>> DimsList;

    // Input and output sizes for allocating memory
    std::vector<int> sizeList;

    // Get the number of input tensors
    std::cout << "ONNX-MODEL SPECIFICATIONS:" << std::endl;
    std::cout << "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾" << std::endl;
    int nbInputs = network->getNbInputs();
    std::cout << "       INPUT       " << std::endl;
    std::cout << "Number of tensors: " << nbInputs << std::endl;

    // Iterate over the input tensors and print their names and dimensions
    for (int i = 0; i < nbInputs; ++i) {
        nvinfer1::ITensor* inputTensor = network->getInput(i);
        std::cout << "Input tensor " << i << ": " << inputTensor->getName() << std::endl;
        std::cout << "Dimensions: ";
        std::vector<int> dimensions;
        for (int j = 0; j < inputTensor->getDimensions().nbDims; ++j) {
            dimensions.push_back(inputTensor->getDimensions().d[j]);
            std::cout << inputTensor->getDimensions().d[j] << " ";
        }
        std::cout << std::endl << std::endl;
        DimsList.emplace_back(inputTensor->getName(), dimensions);
    }

    // Get the number of output tensors
    int nbOutputs = network->getNbOutputs();
    std::cout << "       OUTPUT      " << std::endl;
    std::cout << "Number of tensors: " << nbOutputs << std::endl;

    // Iterate over the output tensors and print their names
    for (int i = 0; i < nbOutputs; ++i) {
        nvinfer1::ITensor* outputTensor = network->getOutput(i);
        std::cout << "Output tensor " << i << ": " << outputTensor->getName() << std::endl;
        std::cout << "Dimensions: ";
        std::vector<int> dimensions;
        for (int j = 0; j < outputTensor->getDimensions().nbDims; ++j) {
            dimensions.push_back(outputTensor->getDimensions().d[j]);
            std::cout << outputTensor->getDimensions().d[j] << " ";
        }
        std::cout << std::endl << std::endl;
        DimsList.emplace_back(outputTensor->getName(), dimensions);
    }

    // Print the contents of the DimsList vector
    std::cout << "TENSOR NAMES AND DIMENSIONS:" << std::endl;
    std::cout << "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾" << std::endl;
    for (const auto& pair : DimsList) {
        std::cout << pair.first << ": ";
        for (const auto& dim : pair.second) {
            std::cout << dim << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    sizeList.resize(DimsList.size(), 1);

    for (unsigned int i = 0; i < DimsList.size(); ++i) {
        for (unsigned int j = 0; j < DimsList[i].second.size(); ++j) {
            sizeList[i] *= abs(DimsList.at(i).second.at(j));
        }
    }

    std::cout << "SIZES DIMENSIONS OF INPUT AND OUTPUT:" << std::endl;
    std::cout << "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾" << std::endl;
    for (unsigned int i = 0; i < DimsList.size(); ++i) {
        std::cout << DimsList.at(i).first << ": " << sizeList[i] << std::endl;
    }
    std::cout << std::endl;

    return sizeList;
}
