#ifndef INT8_ENTROPY_CALIBRATOR_H
#define INT8_ENTROPY_CALIBRATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

#include "NvOnnxParser.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logging.h"
#include "cuda_runtime_api.h"

class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
public:
    Int8EntropyCalibrator(const std::string& calibrationDataPath, const std::string& cacheFile);
    virtual ~Int8EntropyCalibrator();

    int getBatchSize() const noexcept  override { return batchSize_; };
    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept  override;
    const void* readCalibrationCache(size_t& length) noexcept  override;
    void writeCalibrationCache(const void* cache, size_t length) noexcept  override;

    const char* getPathToONNX() const { return pathToONNX; }
    const char * getInputName() const { return inputName; }
    int getInputSize() const { return inputSize; }

    void loadCalibrationDataPublic() { loadCalibrationData(); }


private:
    std::string calibrationDataPath_;
    std::string cacheFile_;

    int batchSize_ = 6;
    int currentBatch_ = 0;
    int inputSize = 640;
    const char * inputName = "images";

    std::vector<char*> imageBuffers_;
    std::vector<std::string> imageFiles_;

    /// For allocating memory (input and outputs in EfficientNMS)
    void *buffers_[5];
    float *input_ = new float[batchSize_ * 3 * inputSize * inputSize * sizeof(float)];      //  input (dynamic_batch x 3 x 512 x 512)
    int *output0_ = new int[batchSize_ * sizeof(int)];                                      //  num_dets
    float *output1_ = new float[batchSize_ * 4 * sizeof(float)];                            //  bboxes
    float *output2_ = new float[batchSize_ * sizeof(float)];                                //  scores
    int *output3_ = new int[batchSize_ * sizeof(int)];                                      //  labels

    /// path to ONNX model
    const char* pathToONNX = "../onnx_model/yolov8m.onnx";

    void loadCalibrationData();
};

#endif // INT8_ENTROPY_CALIBRATOR_H
