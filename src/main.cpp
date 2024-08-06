#include "int8_entropy_calibrator.h"
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>

int main() {

    sample::Logger gLogger_;

    std::string calibrationDataPath = "../data/data.txt";
    std::string cacheFile = "./calibration_data.cache";

    Int8EntropyCalibrator calibrator(calibrationDataPath, cacheFile);
    calibrator.loadCalibrationDataPublic();

    // Create TensorRT builder and network
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger_));
    initLibNvInferPlugins(&gLogger_, "");

    uint32_t flag = 1U <<static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(flag));

    // Parse ONNX model
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger_));
    parser->parseFromFile(calibrator.getPathToONNX(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING));  

    // Create optimization profile
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();
    profile->setDimensions(calibrator.getInputName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{1, 3, calibrator.getInputSize(), calibrator.getInputSize()});
    profile->setDimensions(calibrator.getInputName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{6, 3, calibrator.getInputSize(), calibrator.getInputSize()});
    profile->setDimensions(calibrator.getInputName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{12, 3, calibrator.getInputSize(), calibrator.getInputSize()});

    // Create builder config
    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    config->addOptimizationProfile(profile);
    // size_t workspace_size = (1ULL << 30);
    // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
    // config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1U << 20);

    // Set INT8 mode and calibrator
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setInt8Calibrator(&calibrator);

    // Build engine
    auto engine = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));

    std::ofstream engine_file("../engine/test.engine", std::ios::binary);
    assert(engine_file.is_open() && "Failed to open engine file");
    engine_file.write((char *)engine->data(), engine->size());
    engine_file.close();

    // Clean up
    // delete engine;
    // delete network;
    // delete builder;

    return 0;
}
