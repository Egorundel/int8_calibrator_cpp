#ifndef INT8_ENTROPY_CALIBRATOR_H
#define INT8_ENTROPY_CALIBRATOR_H

#pragma once

#include <opencv2/dnn/dnn.hpp>

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "cuda_utils.h"

#include <string>
#include <vector>
#include <iterator>
#include <dirent.h>
#include <fstream>

#include "auxiliary_utils.h"

class Logger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept {
    // Set log level
    if (severity <= Severity::kINFO) {
      timePrefix();
      std::cout << severityPrefix(severity) << "[TRT] " << std::string(msg) << std::endl;
    }
  }

private:
  static const char *severityPrefix(Severity severity) {
    switch (severity) {
    case Severity::kINTERNAL_ERROR:return "[F] ";
    case Severity::kERROR:return "[E] ";
    case Severity::kWARNING:return "[W] ";
    case Severity::kINFO:return "[I] ";
    case Severity::kVERBOSE:return "[V] ";
    default:
      assert(0);
      return "";
    }
  }
  void timePrefix() {
    std::time_t timestamp = std::time(nullptr);
    tm *tm_local = std::localtime(&timestamp);
    std::cout << "[";
    std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon
              << "/";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
    std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year
              << "-";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
    std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
  }
};

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    Int8EntropyCalibrator(int batchsize, std::vector<int> sizebuffers,int input_w, int input_h, const char* img_dir, const char* calib_cache_name_, const char* input_blob_name, bool read_cache = false);
    virtual ~Int8EntropyCalibrator();
    int getBatchSize() const noexcept override;
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;
    const void* readCalibrationCache(size_t& length) noexcept override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    // batch size of calibration
    int batchsize_;

    // storage of Dims sizes
    std::vector<int> sizebuffers_;

    // image parameters (for example 512x512)
    int input_w_;
    int input_h_;

    // image id, needed to read images
    int img_idx_;

    // directory of images to calibration
    std::string img_dir_;

    // vector of path to images
    std::vector<std::string> img_files_;

    // name of calibration data
    std::string calib_cache_name_;

    // input name of model
    const char* input_blob_name_;

    // flag for reading cache
    bool read_cache_;

    // the vector of calibration values in the calibration file
    std::vector<char> calib_cache_;

    /// For allocating memory
    void* device_input_;
    float* buffer_;
};

#endif // INT8_ENTROPY_CALIBRATOR_H