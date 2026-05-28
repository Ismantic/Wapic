#pragma once

#include <chrono>

#include "model.h"
#include "score.h"

namespace wati {

class Metric {
public:
    int32_t TokenCount;
    int32_t TokenErrors;
    int32_t SentenceCount;
    int32_t SentenceErrors;

    Metric()
        : TokenCount(0), TokenErrors(0)
        , SentenceCount(0), SentenceErrors(0) {}

    float_t GetTokenErrorRate() const {
        return TokenCount > 0 ? (static_cast<float_t>(TokenErrors) / TokenCount * 100.0) : 0.0;
    }

    float_t GetSentenceErrorRate() const {
        return SentenceCount > 0 ? (static_cast<float_t>(SentenceErrors) / SentenceCount * 100.0) : 0.0;
    }
};

class Tester {
public:
    Tester(const Model* model, const Dataset* dataset)
        : scorer_(model), dataset_(dataset) {}

    Metric Run() {
        const int64_t N = static_cast<int64_t>(dataset_->Size());
        std::atomic<int64_t> tok_count{0};
        std::atomic<int64_t> tok_err{0};
        std::atomic<int64_t> sen_count{0};
        std::atomic<int64_t> sen_err{0};

        #pragma omp parallel
        {
            Scorer scorer(scorer_.GetModel());
            std::vector<int64_t> rs;
            int64_t lt = 0, le = 0, ls = 0, lse = 0;
            #pragma omp for schedule(static) nowait
            for (int64_t s = 0; s < N; ++s) {
                const Sentence* sen = dataset_->sens[s];
                const size_t size = sen->Size();
                rs.clear();
                scorer.Viterbi(*sen, rs);
                bool error = false;
                for (size_t t = 0; t < size; ++t) {
                    if (sen->pos[t].label != rs[t]) {
                        le++;
                        error = true;
                    }
                }
                lt += size;
                ls += 1;
                if (error) lse += 1;
            }
            tok_count += lt;
            tok_err += le;
            sen_count += ls;
            sen_err += lse;
        }

        Metric r;
        r.TokenCount = tok_count.load();
        r.TokenErrors = tok_err.load();
        r.SentenceCount = sen_count.load();
        r.SentenceErrors = sen_err.load();
        return r;
    }

private:
    Scorer scorer_;
    const Dataset* dataset_;
};

class Progresser {
public:
    explicit Progresser(const Model* model, size_t window_size, float_t stopeps)
            : model_(model), window_size_(window_size), stopeps_(stopeps)
            , window_pos_(0), window_count_(0), total_time_(0.0) {
        if (window_size_ > 0) {
            error_window_.resize(window_size_, 0.0);
        }
        last_time_ = std::chrono::steady_clock::now();
    }

    bool ShouldStop(float_t current_error) {
        if (window_size_ == 0) {
            return false;
        }

        error_window_[window_pos_] = current_error;
        window_pos_ = (window_pos_ + 1) % window_size_;
        window_count_ ++;

        if (window_count_ >= window_size_) {
            auto [min_error, max_error] = std::minmax_element(
                error_window_.begin(),
                error_window_.end()
            );

            return (*max_error - *min_error) < stopeps_;
        }

        return false;
    }
    void PrintProgress(uint32_t iteration, float_t objective, 
                       uint64_t active_features,
                       float_t token_error, float_t sentence_error,
                       float_t iteration_time) {
        std::cout << "  [" << std::setw(4) << iteration << "] ";

        if (objective >= 0.0) {
            std::cout << "obj=" << std::setw(10) << std::fixed 
                    << std::setprecision(2) << objective << " ";
        } else {
            std::cout << "obj=NA    ";
        }

        std::cout << "act=" << std::setw(8) << active_features << " "
                << "err=" << std::fixed << std::setprecision(2) 
                << std::setw(5) << token_error << "%/" 
                << std::setw(5) << sentence_error << "% "
                << "time=" << std::fixed << std::setprecision(2)
                << iteration_time << "s/"
                << total_time_ << "s" << std::endl;
    }

    bool ReportProgress(uint32_t iteration, float_t objective) {
        Tester tester(model_, model_->GetData());
        Metric r = tester.Run();
        float_t token_error = r.GetTokenErrorRate();
        float_t sentence_error = r.GetSentenceErrorRate();

        auto active_features = model_->CountActiveFeatures();

        auto current_time = std::chrono::steady_clock::now();
        float_t iteration_time = std::chrono::duration<float_t>( 
            current_time - last_time_).count();
        total_time_ += iteration_time;
        last_time_ = current_time;

        PrintProgress(iteration, objective, active_features, 
                      token_error, sentence_error, iteration_time);
        
        if (ShouldStop(token_error)) {
            return false;
        }

        return true;
    }

private:

    const Model* model_;
    size_t window_size_;
    float_t stopeps_;

    std::vector<float_t> error_window_;
    size_t window_pos_;
    size_t window_count_;

    std::chrono::steady_clock::time_point last_time_;
    float_t total_time_;
};

} // namespace wati