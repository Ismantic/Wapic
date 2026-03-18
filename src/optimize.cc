#include "optimize.h"
#include "progress.h"

namespace wati {

void InsertObservation(std::vector<int64_t>& obs, int64_t new_obs) {
    if (std::find(obs.begin(), obs.end(), new_obs) == obs.end()) {
        obs.push_back(new_obs);
    }
}

std::vector<SGDOptimizer::Index> SGDOptimizer::GetIndex() {
    size_t num_sentences = model_->GetData()->Size();
    std::vector<SGDOptimizer::Index> index(num_sentences);
    for (size_t s = 0; s < num_sentences; s++) {
        const Sentence* sen = model_->GetData()->sens[s];
        std::vector<int64_t>& uobs = index[s].unigram_obs;
        std::vector<int64_t>& bobs = index[s].bigram_obs;

        for (size_t t = 0; t < sen->Size(); t++) {
            const Sentence::Pos* pos = &sen->pos[t];

            for (uint32_t p = 0; p < pos->unigram_count; p++) {
                InsertObservation(uobs, pos->unigram_obs[p]);
            }
            for (uint32_t p = 0; p < pos->bigram_count; p++) {
                InsertObservation(bobs, pos->bigram_obs[p]);
            }
        }

        uobs.push_back(static_cast<uint64_t>(-1));
        bobs.push_back(static_cast<uint64_t>(-1));
    }
    return index;
}

void SGDOptimizer::Optimize() {
    auto index = GetIndex();

    size_t num_sentences = model_->GetData()->Size();
    std::vector<uint32_t> sentences(num_sentences);
    std::iota(sentences.begin(), sentences.end(), 0);

    int64_t num_features = model_->FeatureCount();
    std::vector<float_t> gradient(num_features, 0.0);
    std::vector<float_t> penality(num_features, 0.0);

    std::random_device R;
    std::mt19937 G(R());

    float_t total_penality = 0.0;
    auto gradient_state = std::make_unique<GradientState>(model_, gradient);

    int64_t num_labels = model_->LabelCount();

    Progresser p = Progresser(model_, window_size_, stopeps_);
    for (uint32_t iter = 0, i = 0; iter < max_iter_; iter++) {
        std::shuffle(sentences.begin(), sentences.end(), G);

        for (uint32_t s = 0; s < num_sentences; s++, i++) {
            const uint32_t si = sentences[s];
            const Sentence* sen = model_->GetData()->sens[si];
            gradient_state->Compute(*sen);

            const float_t learning_rate = eta_*std::pow(alpha_, static_cast<float_t>(i)/num_sentences);
            total_penality += learning_rate*rho_/num_sentences;

            for (size_t n = 0; n < index[si].unigram_obs.size()-1; n++) {
                int64_t f = model_->GetUnigramIndex(index[si].unigram_obs[n]);
                for (int64_t y = 0; y < num_labels; y++, f++) {
                    weights[f] -= learning_rate*gradient[f];
                    ApplyPenality(f, total_penality, penality);
                    gradient[f] = 0.0;
                }
            }

            for (size_t n = 0; n < index[si].bigram_obs.size()-1; n++) {
                int64_t f = model_->GetBigramIndex(index[si].bigram_obs[n]);
                for (int64_t d = 0; d < num_labels*num_labels; d++, f++) {
                    weights[f] -= learning_rate*gradient[f];
                    ApplyPenality(f, total_penality, penality);
                    gradient[f] = 0.0;
                }
            }
        }

        if (!p.ReportProgress(iter+1, -1.0)) {
            break;
        }
    }
}

void LBFGSOptimizer::Optimize() {
    GradientComputer* G = new GradientComputer(model_, g);
    float_t fx = G->RunGradientComputation(r1, r2);

    Progresser p(model_, window_size, stopeps);

    for(uint32_t k = 0; k < K; k++) {
        if (L1) {
            ComputePseudoGradient();
        }

        ComputeSearchDirection(k);

        if (L1) {
            ConstrainSearchDirection();
        }

        std::copy(x.begin(), x.end(), xp.begin());
        std::copy(g.begin(), g.end(), gp.begin());

        // Line Search (RunGradientComputation)
        if (!PerformLineSearch(k, fx, G)) {
            break;
        }

        if (!p.ReportProgress(k+1, fx)) {
            break;
        }

        UpdateHistory(k);

        if (CheckConvergence(k, fx)) {
            break;
        }
    }

    delete G;
}

void LBFGSOptimizer::ComputePseudoGradient() {
    const float_t rho1 = r1;
    for (int64_t f = 0; f < F; f++) {
        if (x[f] < 0.0)
            pg[f] = g[f] - rho1;
        else if (x[f] > 0.0)
            pg[f] = g[f] + rho1;
        else if (g[f] < -rho1)
            pg[f] = g[f] + rho1;
        else if (g[f] > rho1)
            pg[f] = g[f] - rho1;
        else
            pg[f] = 0.0;
    }
}

void LBFGSOptimizer::ComputeSearchDirection(uint32_t k) {
    // Compute initital search direction
    const float_t* grad_ptr = L1 ? pg.data() : g.data();
    for (int64_t f = 0; f < F; f++) {
        d[f] = - grad_ptr[f];
    }

    if (k == 0) return;


    const uint32_t km = k % M;
    const uint32_t bnd = std::min(k, M);
    std::vector<float_t> alpha(M);

    // Two-loop recursion algorithm
    for (uint32_t i = bnd; i > 0; i--) {
        const uint32_t j = (M + 1 + k - i) % M;
        alpha[i - 1] = p[j] * DotProduct(s[j], d);
        Axpy(-alpha[i - 1], y[j], d);
    }

    // Scale the direction
    const float_t y2 = DotProduct(y[km], y[km]);
    const float_t v = 1.0 / (p[km] * y2);
    for (int64_t f = 0; f < F; f++) {
        d[f] *= v;
    }

    // Second loop of the recursion
    for (uint32_t i = 0; i < bnd; i++) {
        const uint32_t j = (M + k - i) % M;
        float_t beta = p[j] * DotProduct(y[j], d);
        Axpy(alpha[i] - beta, s[j], d);
    }
}

void LBFGSOptimizer::ConstrainSearchDirection() {
    for (int64_t f = 0; f < F; f++) {
        if (d[f] * pg[f] >= 0.0) {
            d[f] = 0.0;
        } 
    }
}

bool LBFGSOptimizer::PerformLineSearch(uint32_t k, float_t& fx, GradientComputer* grd) {
    float_t sc = (k == 0) ? 0.1 : 0.5;
    float_t stp = (k == 0) ? 1.0 / VectorNorm(d) : 1.0;
    float_t gd = L1 ? 0.0 : DotProduct(g, d);
    float_t fi = fx;

    for (uint32_t ls = 1; ; ls++, stp *= sc) {
        // Update position
        for (int64_t f = 0; f < F; f++) {
            x[f] = xp[f] + stp * d[f];
        }

        // Project back to orthant for OWL-QN
        if (L1) {
            ProjectToOrthant();
        }

        fx = grd->RunGradientComputation(r1, r2);

        if (CheckLineSearchConditions(fx, fi, stp, gd, sc)) {
            break;
        }

        if (ls == L) {
            std::cout << "maximum linesearch reached\n";
            std::copy(xp.begin(), xp.end(), x.begin());
            return false;
        }
    }

    return true;
}

void LBFGSOptimizer::ProjectToOrthant() {
    for (int64_t f = 0; f < F; f++) {
        float_t o = xp[f];
        if (o == 0.0) {
            o = -pg[f];
        }
        if (x[f] * o <= 0.0) {
            x[f] = 0.0;
        }
    }
}

bool LBFGSOptimizer::CheckLineSearchConditions(float_t fx, float_t fi, float_t stp, float_t gd, float_t& sc) {
    if (!L1) {
        return CheckWolfeConditions(fx, fi, stp, gd, sc);
    }
    return CheckArmijoRule(fx, fi);
}

bool LBFGSOptimizer::CheckWolfeConditions(float_t fx, float_t fi, float_t stp, float_t gd, float_t& sc) {
    if (fx > fi + stp * gd * 1e-4) {
        sc = 0.5;
        return false;
    } else {
        if (DotProduct(g, d) < gd * 0.9) {
        sc = 2.1;
        return false;
        } else {
        return true;
        }
    }
}

bool LBFGSOptimizer::CheckArmijoRule(float_t fx, float_t fi) {
    float_t vp = 0.0;
    for (int64_t f = 0; f < F; f++) {
        vp += (x[f] - xp[f]) * d[f];
    }
    return fx < fi + vp * 1e-4;
}

void LBFGSOptimizer::UpdateHistory(uint32_t k) {
    const uint32_t kn = (k + 1) % M;
    
    // Update s_k = x_{k+1} - x_k
    for (int64_t f = 0; f < F; f++) {
        s[kn][f] = x[f] - xp[f];
        y[kn][f] = g[f] - gp[f];
    }
    p[kn] = 1.0/DotProduct(y[kn], s[kn]);
}

bool LBFGSOptimizer::CheckConvergence(uint32_t k, float_t fx) {
    const float_t xn = VectorNorm(x);
    const float_t gn = VectorNorm(L1 ? pg : g);
    
    if (gn / std::max(xn, 1.0) <= 1e-5) {
        return true;
    }

    // Check improvement over window
    fh[k % C] = fx;
    if (k >= C) {
        const float_t of = fh[(k + 1) % C];
        float_t dlt = std::abs(of - fx) / of;
        if (dlt < stopeps) {
            return true;
        }
    }

    return false;
}

// Helper functions for vector operations
float_t LBFGSOptimizer::DotProduct(const std::vector<float_t>& a, const std::vector<float_t>& b) {
    float_t sum = 0.0;
    for (int64_t i = 0; i < F; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

void LBFGSOptimizer::Axpy(float_t alpha, const std::vector<float_t>& x, std::vector<float_t>& y) {
    for (int64_t i = 0; i < F; i++) {
        y[i] += alpha * x[i];
    }
}

float_t LBFGSOptimizer::VectorNorm(const std::vector<float_t>& v) {
    return sqrt(DotProduct(v, v));
}

} // namespace wati