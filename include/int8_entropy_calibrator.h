#ifndef INT8_ENTROPY_CALIBRATOR_H
#define INT8_ENTROPY_CALIBRATOR_H

#pragma once

#include <opencv2/dnn/dnn.hpp>

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "cuda_utils.h"
#include "cuda_runtime_api.h"
#include <cuda.h>

#include <string>
#include <vector>
#include <iterator>
#include <dirent.h>
#include <fstream>

#include "yolo_utils.h"

// IInt8EntropyCalibrator2
// IInt8MinMaxCalibrator

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator(int batchsize, std::vector<int> sizebuffers,int input_w, int input_h, const char* img_dir, const char* calib_table_name, const char* input_blob_name, bool read_cache = true);
    virtual ~Int8EntropyCalibrator();
    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    int batchsize_;
    int countbatch_;
    std::vector<int> sizebuffers_;
    int input_w_;
    int input_h_;
    int img_idx_;
    std::string img_dir_;
    std::vector<std::string> img_files_;
    std::string calib_table_name_;
    const char* input_blob_name_;
    bool read_cache_;
    std::vector<char> calib_cache_;

    /// For allocating memory
    void *buffers_[5];
    float *device_input_;
};

#endif // INT8_ENTROPY_CALIBRATOR_H
