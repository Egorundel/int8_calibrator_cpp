#include "auxiliary_utils.h"
// using namespace cv;
// using namespace std;

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
    std::cout << "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾" << std::endl;
    for (unsigned int i = 0; i < DimsList.size(); ++i) {
        std::cout << DimsList.at(i).first << ": " << sizeList[i] << std::endl;
    }
    std::cout << std::endl;

    return sizeList;
}