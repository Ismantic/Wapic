#include "state.h"

#include <algorithm>

namespace wati {

void AtomicIncrement(float_t& value, float_t inc) {
    value += inc;
}

float_t NormalizeVector(float_t* v, uint32_t size) {
    float_t sum = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
        sum += v[i];
    }
    float_t scale = 1.0/sum;
    for (uint32_t i = 0; i < size; ++i) {
        v[i] *= scale;
    }
    return scale;
}

void GradientState::Compute(const Sentence& sen) {
    CheckSize(sen.Size());
    SetBoundary(0, sen.Size()-1);
    ComputePsi(sen);
    ComputeFowardBackward(sen);
    ComputeModelExpectation(sen);
    SubtractEmpirical(sen);
    ComputeLogLoss(sen);
}

void GradientState::CheckSize(uint32_t size) {
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
void GradientState::ComputePsi(const Sentence& sen) {
    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();

    for (size_t t = 0; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        for (int64_t y = 0; y < Y; ++y) {
            float_t sum = 0.0;
            for (uint32_t n = 0; n < pos.unigram_count; ++n) {
                const auto& w = model_->GetUnigramWeights(pos.unigram_obs[n]);
                sum += w[y];
            }
            for (uint32_t yp = 0; yp < Y; ++yp) {
                psi_[(t*Y + yp)*Y + y] = sum;
            }
        }
    }

    for (size_t t = 1; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        for (int64_t yp = 0, d = 0; yp < Y; ++yp) {
            for (int64_t y = 0; y < Y; ++y, ++d) {
                float_t sum = 0.0;
                for (uint32_t n = 0; n < pos.bigram_count; ++n) {
                    const auto& w = model_->GetBigramWeights(pos.bigram_obs[n]);
                    sum += w[d];
                }
                psi_[(t*Y + yp)*Y + y] += sum;
            }
        }
    }

    for (uint32_t i = 0; i < T*Y*Y; ++i) {
        psi_[i] = std::exp(psi_[i]);
    }
}

void GradientState::ComputeFowardBackward(const Sentence& sen) {
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

        unorm_[t] = 1.0/z;
        bnorm_[t] = scale_[t]/z;
    }
}


void GradientState::ComputeModelExpectation(const Sentence& sen) {
    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();

    for (uint32_t t = 0; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        for (int64_t y = 0; y < Y; ++y) {
            // P(y_t = y|x) = alpha[t][y] * beta[t][y] / Z
            float_t e = alpha_[t * Y + y] * beta_[t * Y + y] * unorm_[t];
            for (uint32_t n = 0; n < pos.unigram_count; ++n) {
                const auto o = model_->GetUnigramIndex(pos.unigram_obs[n]);
                AtomicIncrement(gradient_[o + y], e);
            }
        }
    }

    for (uint32_t t = 1; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        for (int64_t yp = 0, d = 0; yp < Y; ++yp) {
            for (int64_t y = 0; y < Y; y++, d++) {
                // P(y_{t-1} = yp, y_t = y|x) = 
                // alpha[t-1][yp] * psi[t][yp][y] * beta[t][y] / Z
                float_t e = alpha_[(t-1) * Y + yp] * 
                            beta_[t * Y + y] *
                            psi_[(t * Y + yp) * Y + y] * 
                            bnorm_[t];
                for (uint32_t n = 0; n < pos.bigram_count; ++n) {
                    auto o = model_->GetBigramIndex(pos.bigram_obs[n]);
                    AtomicIncrement(gradient_[o + d], e);
                }
            }
        }
    }
}


void GradientState::SubtractEmpirical(const Sentence& sen) {
    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();

    for (uint32_t t = 0; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const int64_t y = pos.label;

        for (uint32_t n = 0; n < pos.unigram_count; ++n) {
            const auto& o = model_->GetUnigramIndex(pos.unigram_obs[n]);
            AtomicIncrement(gradient_[o + y], -1.0);
        }
    }

    for (uint32_t t = 1; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const int64_t yp = sen.pos[t-1].label;
        const int64_t y  = pos.label;
        const int64_t d = yp*Y + y;

        for (uint32_t n = 0; n < pos.bigram_count; ++n) {
            const auto& o = model_->GetBigramIndex(pos.bigram_obs[n]);
            AtomicIncrement(gradient_[o + d], -1.0);
        }
    }
}


void GradientState::ComputeLogLoss(const Sentence& sen) {
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
            const auto& w = model_->GetUnigramWeights(pos.unigram_obs[n]);
            pathscore += w[y];
        }
    }

    for (uint32_t t = 1; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const uint32_t yp = sen.pos[t-1].label;
        const uint32_t y = pos.label;
        const uint32_t d = yp * Y + y;
        
        for (uint32_t n = 0; n < pos.bigram_count; ++n) {
            const auto&w = model_->GetBigramWeights(pos.bigram_obs[n]);
            pathscore += w[d];
        }
    }

    logloss_ += logz - pathscore;
}

} // namespace wati