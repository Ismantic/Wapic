#pragma once 

#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <algorithm>
#include <iostream>

#include <math.h>
#include <float.h>
#include <string.h>

#include "model.h"
#include "state.h"

namespace wati {

class SGDOptimizer {
public:
    SGDOptimizer(Model* model, uint32_t max_iter, 
                 size_t window_size, float_t stopeps,
                 float_t eta, float_t alpha, float_t r1)
        : model_(model), max_iter_(max_iter)
        , window_size_(window_size), stopeps_(stopeps)
        , eta_(eta), alpha_(alpha), rho_(r1)
        , weights(model->GetWeights()) {}
    
    void Optimize();

private:
    struct Index {
        std::vector<int64_t> unigram_obs;
        std::vector<int64_t> bigram_obs;
    };
    std::vector<Index> GetIndex();

    Model* model_;
    uint32_t max_iter_;
    size_t window_size_; // stopwin
    float_t stopeps_;
    float_t eta_;
    float_t alpha_;
    float_t rho_;

    std::vector<float_t>& weights;

    inline void ApplyPenality(uint64_t feature_idx, float_t& u, std::vector<float_t>& q) {
        const float_t z = weights[feature_idx];
        if (z > 0.0)
            weights[feature_idx] = std::max(0.0, z-(u+q[feature_idx]));
        else if (z < 0.0)
            weights[feature_idx] = std::min(0.0, z+(u-q[feature_idx]));
        q[feature_idx] += weights[feature_idx] - z;
    }
};

class LBFGSOptimizer {
public:
    LBFGSOptimizer(Model* model, uint32_t window_size, float_t stopeps,
                   uint32_t max_iter, size_t objective_window_size,
                   size_t history_size, size_t max_line_search, float_t r1, float_t r2,
                   uint32_t nthread = 1)
        : model_(model), window_size(window_size), stopeps(stopeps), F(model_->FeatureCount())
        , K(max_iter), C(objective_window_size), M(history_size), L(max_line_search)
        , r1(r1), r2(r2), L1(r1 != 0.0), nthread_(nthread), x(model_->GetWeights()) {
        Init();
    }

    void Optimize();

    // Periodically Save to checkpoint_path_ every checkpoint_every_ iters.
    // checkpoint_every_ = 0 disables.
    void SetCheckpoint(const std::string& path, uint32_t every) {
        checkpoint_path_ = path;
        checkpoint_every_ = every;
    }

private:
    Model* model_;
    std::string checkpoint_path_;
    uint32_t checkpoint_every_ = 0;
    const uint32_t window_size;
    const float_t stopeps;
    const int64_t F;
    const uint32_t K;    // Max iters
    const uint32_t C;    // Window size (objwin)
    const uint32_t M;    // History size
    const uint32_t L;   // MaxLineSearch
    const float_t r1;
    const float_t r2;
    const bool L1;
    const uint32_t nthread_;

    std::vector<float_t>& x;  // Current values (Points to Model->theta)
    std::vector<float_t>  g;  // Current gradient
    std::vector<float_t>  xp; // Previous values
    std::vector<float_t>  gp; // Previous gradient
    std::vector<float_t>  pg; // Pseudo-Gradient (for OWL-QN)
    std::vector<float_t>  d;  // Search Direction
    // L-BFGS history pairs stored as float: the quasi-Newton approximation is
    // tolerant of reduced precision, and this halves the 2*M*F history memory
    // (dot products still accumulate in double).
    std::vector<std::vector<float>> s; // History of Δx
    std::vector<std::vector<float>> y; // History of Δg
    std::vector<float_t> p; // ρ values
    std::vector<float_t> fh; // Function value history

    void Init() {
        g.resize(F);
        xp.resize(F);
        gp.resize(F);
        d.resize(F);

        if (L1) {
            pg.resize(F);
        }

        s.resize(M, std::vector<float>(F));
        y.resize(M, std::vector<float>(F));
        p.resize(M);
        fh.resize(C);
    }

    void ComputePseudoGradient();

    void ComputeSearchDirection(uint32_t k);

    void ConstrainSearchDirection();

    bool PerformLineSearch(uint32_t k, float_t& fx, GradientComputer* grd);

    void ProjectToOrthant();

    bool CheckLineSearchConditions(float_t fx, float_t fi, float_t stp, float_t gd, float_t& sc);

    bool CheckWolfeConditions(float_t fx, float_t fi, float_t stp, float_t gd, float_t& sc);

    bool CheckArmijoRule(float_t fx, float_t fi);

    void UpdateHistory(uint32_t k);

    bool CheckConvergence(uint32_t k, float_t fx);

    // Mixed-precision vector helpers: accept float or double vectors, always
    // accumulate in double.
    template <class VA, class VB>
    float_t DotProduct(const VA& a, const VB& b) {
        float_t sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (int64_t i = 0; i < F; i++) {
            sum += static_cast<float_t>(a[i]) * static_cast<float_t>(b[i]);
        }
        return sum;
    }

    template <class VX>
    void Axpy(float_t alpha, const VX& src, std::vector<float_t>& dst) {
        #pragma omp parallel for schedule(static)
        for (int64_t i = 0; i < F; i++) {
            dst[i] += alpha * static_cast<float_t>(src[i]);
        }
    }

    template <class V>
    float_t VectorNorm(const V& v) {
        return sqrt(DotProduct(v, v));
    }

};

} // namespace wati