#include "state.h"

#include <algorithm>

namespace wati {

float_t NormalizeVector(float_t* v, uint32_t size) {
    float_t sum = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
        sum += v[i];
    }
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        const float_t uniform = 1.0 / static_cast<float_t>(size);
        for (uint32_t i = 0; i < size; ++i) {
            v[i] = uniform;
        }
        return 1.0;
    }
    float_t scale = 1.0/sum;
    for (uint32_t i = 0; i < size; ++i) {
        v[i] *= scale;
    }
    return scale;
}

template <typename GradT>
void GradientState<GradT>::Compute(const Sentence& sen) {
    CheckSize(sen.Size());
    SetBoundary(0, sen.Size()-1);
    ComputePsi(sen);
    ComputeFowardBackward(sen);
    ComputeModelExpectation(sen);
    SubtractEmpirical(sen);
    ComputeLogLoss(sen);
}

template <typename GradT>
void GradientState<GradT>::CheckSize(uint32_t size) {
    if (size == 0 || size <= size_) {
        return;
    }

    const int64_t Y = model_->LabelCount();
    const uint32_t T = size;

    psi_.resize(T*Y*Y);

    alpha_.resize(T*Y);
    beta_.resize(T*Y);

    scale_.resize(T);
    unorm_.resize(T);
    bnorm_.resize(T);

    size_ = size;
}

// 势函数：ψ_t(y', y, x) = exp(Σ λ_k f_k(y', y, x, t))
template <typename GradT>
void GradientState<GradT>::ComputePsi(const Sentence& sen) {
    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();
    const int64_t YY = Y * Y;

    // Unigram contribution: per-position score depends only on current label y,
    // so accumulate once per t and broadcast to all yp slots.
    scratch_y_.resize(static_cast<size_t>(Y));
    float_t* sumY = scratch_y_.data();
    for (size_t t = 0; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const int32_t* uobs = sen.unigram_obs(t);
        for (int64_t y = 0; y < Y; ++y) sumY[y] = 0.0;
        for (uint32_t n = 0; n < pos.unigram_count; ++n) {
            const float_t* w = model_->GetUnigramWeights(uobs[n]);
            for (int64_t y = 0; y < Y; ++y) sumY[y] += w[y];
        }
        float_t* psi_t = psi_.data() + t * YY;
        for (int64_t yp = 0; yp < Y; ++yp) {
            float_t* row = psi_t + yp * Y;
            for (int64_t y = 0; y < Y; ++y) row[y] = sumY[y];
        }
    }

    // Bigram contribution: full Y*Y matrix per t.
    scratch_yy_.resize(static_cast<size_t>(YY));
    float_t* sumYY = scratch_yy_.data();
    for (size_t t = 1; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const int32_t* bobs = sen.bigram_obs(t);
        for (int64_t d = 0; d < YY; ++d) sumYY[d] = 0.0;
        for (uint32_t n = 0; n < pos.bigram_count; ++n) {
            const float_t* w = model_->GetBigramWeights(bobs[n]);
            for (int64_t d = 0; d < YY; ++d) sumYY[d] += w[d];
        }
        float_t* psi_t = psi_.data() + t * YY;
        for (int64_t d = 0; d < YY; ++d) psi_t[d] += sumYY[d];
    }

    for (uint32_t i = 0; i < T*Y*Y; ++i) {
        psi_[i] = std::exp(psi_[i]);
    }
}

template <typename GradT>
void GradientState<GradT>::ComputeFowardBackward(const Sentence& sen) {
    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();

    // 1. Forward
    // t = 0
    for (int64_t y = 0; y < Y; ++y) {
        alpha_[y] = psi_[y];
    }

    scale_[0] = NormalizeVector(alpha_.data(), Y);

    for (uint32_t t = 1; t < end_ + 1; ++t) {
        for (int64_t y = 0; y < Y; ++y) {
            float_t sum = 0.0;
            for (int64_t yp = 0; yp < Y; ++yp) {
                // alpha[t][y] = Σ(alpha[t-1][yp] * psi[t][yp][y])
                sum += alpha_[(t-1)*Y + yp] * psi_[(t*Y + yp)*Y + y];
            }
            alpha_[t*Y + y] = sum;
        }
        scale_[t] = NormalizeVector(alpha_.data() + t*Y, Y);
    }

    // 2. Backward
    // t = T-1
    for (int64_t yp = 0; yp < Y; ++yp) {
        beta_[(T-1)*Y + yp] = 1.0/Y;
    }
    for (uint32_t t = T - 1; t > start_; --t) {
        for (int64_t yp = 0; yp < Y; ++yp) {
            float_t sum = 0.0;
            for (int64_t y = 0; y < Y; ++y) {
                // beta[t-1][y] = Σ(beta[t][y] * psi[t][yp][y])
                sum += beta_[t*Y + y] * psi_[(t*Y + yp)*Y + y];
            }
            beta_[(t-1)*Y + yp] = sum;
        }
        NormalizeVector(beta_.data()+(t-1)*Y, Y);
    }

    // 3.
    for (uint32_t t = 0; t < T; ++t) {
        float_t z = 0.0;
        // Z = Σ(alpha[t][y] * beta[t][y])
        for (int64_t y = 0; y < Y; ++y) {
            z += alpha_[t*Y + y] * beta_[t*Y + y];
        }
        if (!(z > 0.0) || !std::isfinite(z)) {
            z = 1.0;
        }

        unorm_[t] = 1.0/z;
        bnorm_[t] = scale_[t]/z;
    }
}


template <typename GradT>
void GradientState<GradT>::ComputeModelExpectation(const Sentence& sen) {
    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();
    const int64_t YY = Y * Y;

    // Unigram block: precompute e[y] per t, then write contiguous gradient[o..o+Y]
    scratch_y_.resize(static_cast<size_t>(Y));
    float_t* eY = scratch_y_.data();
    for (uint32_t t = 0; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const int32_t* uobs = sen.unigram_obs(t);
        const float_t un = unorm_[t];
        for (int64_t y = 0; y < Y; ++y) {
            eY[y] = alpha_[t * Y + y] * beta_[t * Y + y] * un;
        }
        for (uint32_t n = 0; n < pos.unigram_count; ++n) {
            const int64_t o = model_->GetUnigramIndex(uobs[n]);
            GradT* g = gradient_.data() + o;
            for (int64_t y = 0; y < Y; ++y) g[y] += eY[y];
        }
    }

    // Bigram block: precompute e[d=yp*Y+y] per t, then write contiguous gradient[o..o+YY]
    scratch_yy_.resize(static_cast<size_t>(YY));
    float_t* eYY = scratch_yy_.data();
    for (uint32_t t = 1; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const int32_t* bobs = sen.bigram_obs(t);
        const float_t bn = bnorm_[t];
        for (int64_t yp = 0, d = 0; yp < Y; ++yp) {
            const float_t alpha_yp = alpha_[(t-1) * Y + yp];
            for (int64_t y = 0; y < Y; ++y, ++d) {
                eYY[d] = alpha_yp * beta_[t * Y + y] *
                         psi_[(t * Y + yp) * Y + y] * bn;
            }
        }
        for (uint32_t n = 0; n < pos.bigram_count; ++n) {
            const int64_t o = model_->GetBigramIndex(bobs[n]);
            GradT* g = gradient_.data() + o;
            for (int64_t d = 0; d < YY; ++d) g[d] += eYY[d];
        }
    }
}


template <typename GradT>
void GradientState<GradT>::SubtractEmpirical(const Sentence& sen) {
    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();

    for (uint32_t t = 0; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const int64_t y = pos.label;

        for (uint32_t n = 0; n < pos.unigram_count; ++n) {
            const auto& o = model_->GetUnigramIndex(sen.unigram_obs(t)[n]);
            gradient_[o + y] += -1.0;
        }
    }

    for (uint32_t t = 1; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const int64_t yp = sen.pos[t-1].label;
        const int64_t y  = pos.label;
        const int64_t d = yp*Y + y;

        for (uint32_t n = 0; n < pos.bigram_count; ++n) {
            const auto& o = model_->GetBigramIndex(sen.bigram_obs(t)[n]);
            gradient_[o + d] += -1.0;
        }
    }
}


template <typename GradT>
void GradientState<GradT>::ComputeLogLoss(const Sentence& sen) {
    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();

    float_t logz = 0.0;
    for (int64_t y = 0; y < Y; ++y) {
        logz += alpha_[(T-1)*Y + y];
    }
    logz = std::log(logz);

    for (uint32_t t = 0; t < T; ++t) {
        logz -= std::log(scale_[t]);
    }

    float_t pathscore = 0.0;
    for (uint32_t t = 0; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const int64_t y = pos.label;

        for (uint32_t n = 0; n < pos.unigram_count; ++n) {
            const auto& w = model_->GetUnigramWeights(sen.unigram_obs(t)[n]);
            pathscore += w[y];
        }
    }

    for (uint32_t t = 1; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const uint32_t yp = sen.pos[t-1].label;
        const uint32_t y = pos.label;
        const uint32_t d = yp * Y + y;

        for (uint32_t n = 0; n < pos.bigram_count; ++n) {
            const auto&w = model_->GetBigramWeights(sen.bigram_obs(t)[n]);
            pathscore += w[d];
        }
    }

    logloss_ += logz - pathscore;
}

// Explicit instantiations: float for GradientComputer's per-thread scratch
// gradients, double for the SGD path's shared gradient.
template class GradientState<float>;
template class GradientState<double>;

} // namespace wati
