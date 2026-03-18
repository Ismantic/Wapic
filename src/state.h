#pragma once 

#include <vector>
#include <memory>
#include <limits>
#include <iostream>

#include <stdint.h>
#include <math.h>

#include "model.h"
#include "sentence.h"

namespace wati {

class GradientState {
public:
    GradientState(const Model* model, std::vector<float_t>& gradient)
        : model_(model)
        , gradient_(gradient)
        , logloss_(0.0)
        , size_(0)
        , start_(0)
        , end_(0) { }

    GradientState(const GradientState&) = delete;
    GradientState& operator=(const GradientState&) = delete; 

    GradientState(GradientState&&) = delete; 
    GradientState& operator=(GradientState&&) = delete; 

    void Compute(const Sentence& sen);
    void ResetLoss() { logloss_ = 0.0; }
    float_t GetLoss() { return logloss_; }
    float_t* GetGradients() { return gradient_.data(); }

private:
    const Model* model_;
    std::vector<float_t>& gradient_;

    float_t logloss_;
    uint32_t size_;
    uint32_t start_;
    uint32_t end_;

    std::vector<float_t> psi_;
    std::vector<float_t> alpha_;
    std::vector<float_t> beta_;
    std::vector<float_t> scale_;
    std::vector<float_t> unorm_;
    std::vector<float_t> bnorm_;


    void ComputePsi(const Sentence& sen);
    void ComputeFowardBackward(const Sentence& sen);
    void ComputeModelExpectation(const Sentence& sen);
    void SubtractEmpirical(const Sentence& sen);
    void ComputeLogLoss(const Sentence& sen);

    void CheckSize(uint32_t size);
    void SetBoundary(uint32_t s, uint32_t e) {
        start_ = s;
        end_ = e;
    }

    const float_t* GetPsi() const { return psi_.data(); }
    const float_t* GetAlpha() const { return alpha_.data(); }
    const float_t* GetBeta() const { return beta_.data(); }
    const float_t* GetUnigramNorm() const { return unorm_.data(); }
    const float_t* GetBigramNorm() const { return bnorm_.data(); }
};

class GradientComputer {
public:
    GradientComputer(const Model* model, std::vector<float_t>& gradient)
        : model_(model) 
        , gradient_state_(std::make_unique<GradientState>(model, gradient)) {}
    
    float_t RunGradientComputation(float_t r1, float_t r2) {
        const Model* m = model_;
        const int64_t feature_count = m->FeatureCount();
        const std::vector<float_t>& W = m->GetWeights();

        float_t* G = gradient_state_->GetGradients();
        for (int64_t i = 0; i < feature_count; ++i) {
            G[i] = 0.0;
        }

        // Process all Samples sequentially
        const auto& data = m->GetData();
        const auto& samples = data->sens;
        gradient_state_->ResetLoss();

        for (const auto& sample : samples) {
            gradient_state_->Compute(*sample);
        }

        float_t fx = gradient_state_->GetLoss();

        float_t n1 = 0.0, n2 = 0.0;
        for (int64_t f = 0; f < feature_count; ++f) {
            const float_t v = W[f];
            G[f] = G[f] + r2*v;
            n1 += std::abs(v);
            n2 += v*v;
        }

        fx += n1*r1 + n2*r2/2.0;
        return fx;
    }

private:
    const Model* model_;
    std::unique_ptr<GradientState> gradient_state_;
};

} // namespace wati