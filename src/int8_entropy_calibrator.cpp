#include "int8_entropy_calibrator.h"
#include <iostream>
#include <fstream>

Int8EntropyCalibrator::Int8EntropyCalibrator(const std::string& calibrationDataPath, const std::string& cacheFile)
    : calibrationDataPath_(calibrationDataPath), cacheFile_(cacheFile) {
    cudaMalloc((void**)&input_, batchSize_ * 3 * inputSize * inputSize * sizeof(float));
    cudaMalloc((void**)&output0_, batchSize_ * sizeof(int));
    cudaMalloc((void**)&output1_, batchSize_ * 4 * sizeof(float));
    cudaMalloc((void**)&output2_, batchSize_ * sizeof(float));
    cudaMalloc((void**)&output3_, batchSize_ * sizeof(int));
}

Int8EntropyCalibrator::~Int8EntropyCalibrator() {
    for (auto& buffer : imageBuffers_) {
        delete[] buffer;
    }
    imageBuffers_.clear();

    cudaFree(input_);
    cudaFree(output0_);
    cudaFree(output1_);
    cudaFree(output2_);
    cudaFree(output3_);
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if (currentBatch_ < batchSize_) {
        for (int i = 0; i < nbBindings; ++i) {
            bindings[i] = imageBuffers_[currentBatch_];
        }
        currentBatch_++;
        return true;
    }
    return false;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept {
    std::ifstream cacheFile(cacheFile_, std::ios::binary | std::ios::ate);
    if (!cacheFile.is_open()) {
        return nullptr;
    }
    length = cacheFile.tellg();
    cacheFile.seekg(0, std::ios::beg);
    char* data = new char[length];
    cacheFile.read(data, length);
    cacheFile.close();
    return data;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept {
    std::ofstream cacheFile(cacheFile_, std::ios::binary);
    cacheFile.write(reinterpret_cast<const char*>(cache), length);
    cacheFile.close();
}

void Int8EntropyCalibrator::loadCalibrationData() {
    std::ifstream fileList(calibrationDataPath_);
    if (!fileList.is_open()) {
        std::cerr << "Error: Could not open calibration data file." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(fileList, line)) {
        imageFiles_.push_back(line);
    }
    fileList.close();

    for (const auto& file : imageFiles_) {
        cv::Mat image = cv::imread(file);
        if (image.empty()) {
            std::cerr << "Error: Could not read image " << file << std::endl;
            continue;
        }

        // Resize image to 512x512
        cv::resize(image, image, cv::Size(inputSize, inputSize));

        // Convert to float and normalize
        cv::Mat floatImage;
        image.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

        // Allocate buffer for the image
        char* buffer = new char[inputSize * inputSize * 3 * sizeof(float)];
        floatImage.convertTo(cv::Mat(inputSize, inputSize, CV_32FC3, buffer), CV_32FC3);
        imageBuffers_.push_back(buffer);
    }
}