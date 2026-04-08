#pragma once

#include <vector>
#include <memory>
#include <limits>
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

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

// Parallel gradient computation with thread pool.
//
// Uses per-thread gradient copies to avoid contention, with two-phase
// parallel execution: (1) dynamic job scheduling for forward-backward
// computation, (2) parallel reduction of per-thread gradients.
class GradientComputer {
    static constexpr uint32_t BATCH_SIZE = 16;
public:
    GradientComputer(const Model* model, std::vector<float_t>& gradient, uint32_t nthread = 1)
        : model_(model), gradient_(gradient), nthread_(std::max(nthread, 1u))
        , stop_(false), generation_(0), compute_done_(0), reduce_done_(0)
        , job_next_(0), r1_(0), r2_(0)
    {
        local_grads_.resize(nthread_);
        partial_n1_.resize(nthread_, 0.0);
        partial_n2_.resize(nthread_, 0.0);
        for (uint32_t w = 0; w < nthread_; w++) {
            local_grads_[w].resize(model->FeatureCount(), 0.0);
            states_.push_back(std::make_unique<GradientState>(model, local_grads_[w]));
        }

        for (uint32_t w = 1; w < nthread_; w++)
            workers_.emplace_back([this, w]() { WorkerLoop(w); });
    }

    ~GradientComputer() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_start_.notify_all();
        for (auto& t : workers_) t.join();
    }

    GradientComputer(const GradientComputer&) = delete;
    GradientComputer& operator=(const GradientComputer&) = delete;

    float_t RunGradientComputation(float_t r1, float_t r2) {
        const int64_t F = model_->FeatureCount();
        const auto& samples = model_->GetData()->sens;
        num_samples_ = samples.size();
        const auto& W = model_->GetWeights();

        // Single-threaded fast path
        if (nthread_ == 1) {
            auto& g = local_grads_[0];
            std::fill(g.begin(), g.end(), 0.0);
            states_[0]->ResetLoss();
            for (size_t i = 0; i < num_samples_; i++)
                states_[0]->Compute(*samples[i]);

            float_t fx = states_[0]->GetLoss();
            float_t n1 = 0.0, n2 = 0.0;
            for (int64_t f = 0; f < F; f++) {
                const float_t v = W[f];
                gradient_[f] = g[f] + r2 * v;
                n1 += std::abs(v);
                n2 += v * v;
            }
            return fx + n1 * r1 + n2 * r2 / 2.0;
        }

        // Multi-threaded: store r1/r2 for reduce phase
        r1_ = r1; r2_ = r2;

        // Reset counters and wake workers
        job_next_.store(0, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lk(mu_);
            compute_done_ = 0;
            reduce_done_ = 0;
            generation_++;
        }
        cv_start_.notify_all();

        // Main thread participates in compute + reduce
        RunWorker(0);

        // Wait for all worker reduce phases to complete
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_done_.wait(lk, [this]() { return reduce_done_ == nthread_ - 1; });
        }

        // Sum per-thread losses and norms
        float_t fx = 0.0;
        float_t n1 = 0.0, n2 = 0.0;
        for (uint32_t w = 0; w < nthread_; w++) {
            fx += states_[w]->GetLoss();
            n1 += partial_n1_[w];
            n2 += partial_n2_[w];
        }
        return fx + n1 * r1 + n2 * r2 / 2.0;
    }

private:
    const Model* model_;
    std::vector<float_t>& gradient_;
    uint32_t nthread_;
    std::vector<std::vector<float_t>> local_grads_;
    std::vector<std::unique_ptr<GradientState>> states_;
    std::vector<float_t> partial_n1_;
    std::vector<float_t> partial_n2_;

    // Thread pool synchronization
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable cv_start_;
    std::condition_variable cv_done_;
    std::condition_variable cv_barrier_;
    bool stop_;
    uint64_t generation_;
    uint32_t compute_done_;
    uint32_t reduce_done_;

    // Dynamic job scheduling
    std::atomic<size_t> job_next_;
    size_t num_samples_ = 0;
    float_t r1_, r2_;

    void RunWorker(uint32_t w) {
        const int64_t F = model_->FeatureCount();
        const auto& samples = model_->GetData()->sens;
        const auto& W = model_->GetWeights();

        // Phase 1: Compute per-thread gradients via dynamic job scheduling
        std::fill(local_grads_[w].begin(), local_grads_[w].end(), 0.0);
        states_[w]->ResetLoss();
        while (true) {
            size_t pos = job_next_.fetch_add(BATCH_SIZE, std::memory_order_relaxed);
            if (pos >= num_samples_) break;
            size_t end = std::min(pos + BATCH_SIZE, num_samples_);
            for (size_t i = pos; i < end; i++)
                states_[w]->Compute(*samples[i]);
        }

        // Barrier: wait for all threads to finish compute phase
        {
            std::unique_lock<std::mutex> lk(mu_);
            compute_done_++;
            if (compute_done_ == nthread_)
                cv_barrier_.notify_all();
            else
                cv_barrier_.wait(lk, [this]() { return compute_done_ == nthread_; });
        }

        // Phase 2: Parallel reduce — each thread merges its slice of F
        int64_t f_lo = F * w / nthread_;
        int64_t f_hi = F * (w + 1) / nthread_;
        float_t n1 = 0.0, n2 = 0.0;
        for (int64_t f = f_lo; f < f_hi; f++) {
            float_t g_sum = local_grads_[0][f];
            for (uint32_t t = 1; t < nthread_; t++)
                g_sum += local_grads_[t][f];
            const float_t v = W[f];
            gradient_[f] = g_sum + r2_ * v;
            n1 += std::abs(v);
            n2 += v * v;
        }
        partial_n1_[w] = n1;
        partial_n2_[w] = n2;

        // Workers signal completion; main thread returns directly
        if (w != 0) {
            std::lock_guard<std::mutex> lk(mu_);
            reduce_done_++;
            cv_done_.notify_one();
        }
    }

    void WorkerLoop(uint32_t w) {
        uint64_t last_gen = 0;
        while (true) {
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_start_.wait(lk, [&]() { return stop_ || generation_ > last_gen; });
                if (stop_) return;
                last_gen = generation_;
            }
            RunWorker(w);
        }
    }
};

} // namespace wati
