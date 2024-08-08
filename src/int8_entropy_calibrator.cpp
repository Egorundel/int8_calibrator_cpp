#include "int8_entropy_calibrator.h"

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

Int8EntropyCalibrator::Int8EntropyCalibrator(int batchsize, std::vector<int> sizebuffers, int input_w, int input_h, const char* img_dir, const char* calib_table_name,
                                               const char* input_blob_name, bool read_cache)
    : batchsize_(batchsize)
    , countbatch_(0)
    , sizebuffers_(sizebuffers)
    , input_w_(input_w)
    , input_h_(input_h)
    , img_idx_(0)
    , img_dir_(img_dir)
    , calib_table_name_(calib_table_name)
    , input_blob_name_(input_blob_name)
    , read_cache_(read_cache)
{
    device_input_ = new float[batchsize_ * sizebuffers_[0]];

    /// Allocate memory and buffers_ for the input and output tensors
    for (unsigned int i = 0; i < sizebuffers_.size(); ++i)
    {
        CUDA_CHECK(cudaMalloc(&buffers_[i], batchsize_ * sizebuffers_[i] * sizeof(float)));
    }
    
    read_files_in_dir(img_dir, img_files_);
}

Int8EntropyCalibrator::~Int8EntropyCalibrator()
{
    for (unsigned int i = 0; i < sizebuffers_.size(); i++)
    {
        CUDA_CHECK(cudaFree(buffers_[i]));
    }
    delete[] device_input_;
}

int Int8EntropyCalibrator::getBatchSize() const noexcept
{
    return batchsize_;
}

bool Int8EntropyCalibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    clock_t start = 0, end = 0;
    
    if (img_idx_ + batchsize_ > (int)img_files_.size()) {
        return false;
    }

    std::vector<cv::Mat> input_imgs_;
    for (int i = img_idx_; i < img_idx_ + batchsize_; i++) {
        
        // std::cout << img_files_[i] << "  " << i << std::endl;
        start = clock();
        cv::Mat temp = cv::imread(img_dir_ + img_files_[i]);
        if (temp.empty()){
            std::cerr << "Fatal error: image cannot open!" << std::endl;
            return false;
        }
        cv::Mat LetterBoxImg;
        cv::Vec4d params;
	    LetterBox(temp, LetterBoxImg, params, cv::Size(input_w_, input_h_));
        input_imgs_.push_back(LetterBoxImg);  
        end = clock();
    }
    std::cout << "Calibrated batch " << countbatch_ << " in " << ((double)(end-start)/CLOCKS_PER_SEC) * 1000 << " seconds" << std::endl;
    countbatch_ += 1;
    img_idx_ += batchsize_;
    
    CUDA_CHECK(cudaMemcpy(buffers_[0], device_input_, batchsize_ * sizebuffers_[0] * sizeof(float), cudaMemcpyHostToDevice));
    
    // assert(!strcmp(names[0], input_blob_name_));
    // bindings[0] = device_input_;
    bindings[0] = buffers_[0];
    return true;
}

const void* Int8EntropyCalibrator::readCalibrationCache(size_t& length) noexcept
{
    std::cout << "READING CALIB CACHE" << std::endl;
    std::cout << "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾" << std::endl;
    std::cout << calib_table_name_ << std::endl;
    calib_cache_.clear();
    std::ifstream input(calib_table_name_, std::ios::binary);
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
    std::cout << "WRITING CALIB CACHE" << std::endl;
    std::cout << "‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾" << std::endl;
    std::cout << calib_table_name_ << " size: " << length << std::endl;

    assert(!calib_table_name_.empty());
    std::ofstream output(calib_table_name_, std::ios::binary);
    // output.write(reinterpret_cast<const char*>(cache), length);
    output.write((const char*)cache, length);
    output.close();
}
