#include "int8_entropy_calibrator.h"

Int8EntropyCalibrator::Int8EntropyCalibrator(int batchsize, std::vector<int> sizebuffers, int input_w, int input_h, const char* img_dir, const char* calib_cache_name,
                                               const char* input_blob_name, bool read_cache): 
                                               batchsize_(batchsize),
                                               sizebuffers_(sizebuffers),
                                               input_w_(input_w),
                                               input_h_(input_h),
                                               img_idx_(0),
                                               img_dir_(img_dir),
                                               calib_cache_name_(calib_cache_name),
                                               input_blob_name_(input_blob_name),
                                               read_cache_(read_cache)
{
    
    CUDA_CHECK(cudaMalloc(&device_input_, batchsize_ * sizebuffers_[0] * sizeof(float)));
    
    read_files_in_dir(img_dir, img_files_);
}

Int8EntropyCalibrator::~Int8EntropyCalibrator()
{
    CUDA_CHECK(cudaFree(device_input_));
}

int Int8EntropyCalibrator::getBatchSize() const noexcept
{
    return batchsize_;
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept {
    if (img_idx_ + batchsize_ > (int)img_files_.size()) {
        return false;
    }

    std::vector<cv::Mat> input_imgs_;
    for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
        // std::cout << img_files_[i] << "  " << i << std::endl;
        cv::Mat temp = cv::imread(img_dir_ + img_files_[i]);
        if (temp.empty()) {
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        cv::Mat pr_img = preprocess_img(temp, input_w_, input_h_);
        input_imgs_.push_back(pr_img);
    }
    img_idx_ += batchsize_;
    cv::Mat blob = cv::dnn::blobFromImages(input_imgs_, 1.0 / 255.0, cv::Size(input_w_, input_h_), cv::Scalar(0, 0, 0),
                                           true, false);

    CUDA_CHECK(cudaMemcpy(device_input_, blob.ptr<float>(0), batchsize_ * sizebuffers_[0] * sizeof(float), cudaMemcpyHostToDevice));
    assert(!strcmp(names[0], input_blob_name_));
    bindings[0] = device_input_;
    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept
{
    calib_cache_.clear();
    std::ifstream input(calib_cache_name_, std::ios::binary);
    input >> std::noskipws;
    if (read_cache_ && input.good())
    {
        std::copy(std::istream_iterator<char>(input), std::istream_iterator<char>(), std::back_inserter(calib_cache_));
    }
    length = calib_cache_.size();

    if (length) {
        std::cout << "Using cached calibration table to build the engine" << std::endl;
    } else {
        std::cout << "New calibration table will be created to build the engine" << std::endl;
        std::cout << "Images for calibration: " << (int)img_files_.size() << std::endl;
    }

    return length ? calib_cache_.data() : nullptr;
}

void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    std::cout << "Size of calibration file \"" << calib_cache_name_ << "\": " << length << std::endl;

    assert(!calib_cache_name_.empty());
    std::ofstream output(calib_cache_name_, std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
    // output.write((const char*)cache, length);
    output.close();
}