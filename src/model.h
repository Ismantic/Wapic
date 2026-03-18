#pragma once

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <algorithm>

#include <stdint.h>

#include "sentence.h"
#include "data.h"

namespace wati {

class Model {
private:
    // Data
    std::unique_ptr<DataProcessor> processor_;
    std::unique_ptr<Dataset> data_;
    std::unique_ptr<Dataset> test_data_;

    // Size
    int64_t label_count_; // Y 
    int64_t observation_count_; // O
    int64_t feature_count_; // F

    // Model Parameters
    std::vector<char> kind_; // [O] observation type : unigram/bigram
    std::vector<int64_t> uoff_; // [O] unigram weights offset
    std::vector<int64_t> boff_; // [O] bigram weights offset
    std::vector<float_t> theta_; // [F] feature weights

public:
    void LoadPatterns(const std::string& filename) {
        processor_->LoadPatterns(filename);
    }

    void LoadData(const std::string& filename) {
        std::ifstream file(filename);
        data_ = std::unique_ptr<Dataset>(processor_->LoadDataset(file, true));
    }

    void LoadTestData(const std::string& filename) {
        std::ifstream file(filename);
        test_data_ = std::unique_ptr<Dataset>(processor_->LoadDataset(file, true));
    }

    const DataProcessor* GetDataProcessor() const { return processor_.get(); }
    const Dataset* GetData() const { return data_.get(); }
    const Dataset* GetTestData() const { return test_data_.get(); }

    int64_t LabelCount() const { return label_count_; }
    int64_t ObservationCount() const { return observation_count_; }
    int64_t FeatureCount() const { return feature_count_; }
    int64_t ModelSize() const { return theta_.size(); }

   const float_t* GetUnigramWeights(int64_t n) const {
        const float_t* w = theta_.data() + uoff_[n];
        return w;
    }
    float_t* GetUnigramWeights(int64_t n) {
        float_t* w = theta_.data() + uoff_[n];
        return w;
    }
    int64_t GetUnigramIndex(int64_t n) const {
        return uoff_[n];
    }
 
    const float_t* GetBigramWeights(int64_t n) const {
        const float_t* w = theta_.data() + boff_[n];
        return w;
    }
    float_t* GetBigramWeights(int64_t n) {
        float_t* w = theta_.data() + boff_[n];
        return w;
    }
    int64_t GetBigramIndex(int64_t n) const {
        return boff_[n];
    }

    const std::vector<float_t>& GetWeights() const { return theta_; }
    std::vector<float_t>& GetWeights() { return theta_; }

    int64_t CountActiveFeatures() const {
        return std::count_if(theta_.begin(), theta_.end(),
                             [](float_t p) { return p != 0.0;});
    }


    Model(std::unique_ptr<DataProcessor> processor = nullptr)
        : processor_(std::move(processor))
        , data_(nullptr)
        , test_data_(nullptr)
        , label_count_(0)
        , observation_count_(0)
        , feature_count_(0)
    {}

    ~Model() = default;

    // Prevent copying 
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete; 

    // Allow moving 
    Model(Model&&) = default; 
    Model& operator=(Model&&) = default; 

    void Sync();

    void Save(const std::string& filename) const;
    void Load(const std::string& filename);        
};

} // namespace wati