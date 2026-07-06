#include "score.h"

#include <numeric>
#include <algorithm>
#include <math.h>

namespace wati {

bool Scorer::ComputeScore(const Sentence& sen) {
    if (!model_) return false;
    ResizePsi(sen.Size(), model_->LabelCount());
    ComputeUnigramScores(sen);
    ComputeBigramScores(sen);
    return true;
}

void Scorer::ResizePsi(size_t T, int64_t Y) {
    psi_.resize(T);
    for (auto& m : psi_) {
        m.resize(Y);
        for (auto& r : m) {
            r.resize(Y);
        }
    }
}

void Scorer::ComputeUnigramScores(const Sentence& sen) {
    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();

    // Loop order n,y so each feature's weight slice is loaded once.
    // sum_y is per-position accumulator across features.
    sum_y_.resize(static_cast<size_t>(Y));
    for (uint32_t t = 0; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const int32_t* obs = sen.unigram_obs(t);
        for (int64_t y = 0; y < Y; ++y) sum_y_[y] = 0.0;
        for (uint32_t n = 0; n < pos.unigram_count; ++n) {
            int32_t o = obs[n];
            if (model_->IsUnigramDead(o)) continue;
            const auto w = model_->GetUnigramWeights(o);
            for (int64_t y = 0; y < Y; ++y) sum_y_[y] += w[y];
        }
        // Broadcast: psi_[t][yp][y] = sum_y[y]
        for (int64_t yp = 0; yp < Y; ++yp) {
            auto& row = psi_[t][yp];
            for (int64_t y = 0; y < Y; ++y) row[y] = sum_y_[y];
        }
    }
}

void Scorer::ComputeBigramScores(const Sentence& sen) {
    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();
    const size_t YY = static_cast<size_t>(Y) * static_cast<size_t>(Y);

    sum_d_.resize(YY);
    for (uint32_t t = 1; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        const int32_t* obs = sen.bigram_obs(t);
        for (size_t d = 0; d < YY; ++d) sum_d_[d] = 0.0;
        for (uint32_t n = 0; n < pos.bigram_count; ++n) {
            int32_t o = obs[n];
            if (model_->IsBigramDead(o)) continue;
            const auto w = model_->GetBigramWeights(o);
            for (size_t d = 0; d < YY; ++d) sum_d_[d] += w[d];
        }
        // Add to psi: psi_[t][yp][y] += sum_d[yp*Y+y]
        for (int64_t yp = 0, d = 0; yp < Y; ++yp) {
            auto& row = psi_[t][yp];
            for (int64_t y = 0; y < Y; ++y, ++d) row[y] += sum_d_[d];
        }
    }
}

void Scorer::Viterbi(const Sentence& sen, 
                     std::vector<int64_t>& labels,
                     float_t* score,
                     std::vector<float_t>* path_scores) {
    ComputeScore(sen);

    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();

    BackData va(T, Y);

    for (int64_t y = 0; y < Y; y++) {
        va.curr[y] = psi_[0][0][y];
    }

    for (size_t t = 1; t < T; t++) {
        va.prev = va.curr;

        for (int64_t y = 0; y < Y; y++) {
            float_t score = -std::numeric_limits<float_t>::infinity();
            int64_t prev = 0;

            for (int64_t p = 0; p < Y; p++) {
                float_t value = va.prev[p] + psi_[t][p][y];
                if (value > score) {
                    score = value;
                    prev = p;
                }
            }
            va.back[t][y] = prev;
            va.curr[y] = score;
        }
    }

    labels.resize(T);

    int64_t new_label = 0;
    for (int64_t y = 1; y < Y; y++) {
        if (va.curr[y] > va.curr[new_label]) {
            new_label = y;
        }
    }


    if (score != nullptr) {
        *score = va.curr[new_label];
    }

    if (path_scores != nullptr) {
        path_scores->resize(T);
    }

    for (size_t t = T; t > 0; t--) {
        const int64_t p = (t != 1) ? va.back[t-1][new_label] : 0;
        const int64_t y = new_label;
        labels[t-1] = y;

        if (path_scores != nullptr) {
            (*path_scores)[t-1] = psi_[t-1][p][y];
        }

        new_label = p;
    }
}

void Scorer::LabelSentences(std::istream& in, std::ostream& out) {
    const DataProcessor* processor = model_->GetDataProcessor();
    const size_t BATCH = 256;

    struct Result {
        std::vector<int64_t> labels;
        std::vector<float_t> path_scores;
        float_t score = 0.0;
        bool valid = false;
    };

    while (true) {
        // Read a batch of sentences serially (file I/O).
        std::vector<Sentence*> sens;
        sens.reserve(BATCH);
        while (sens.size() < BATCH) {
            RawStrs* raw = processor->ReadRawStrs(in);
            if (raw == nullptr) break;
            Sentence* sen = processor->RawToSentence(raw, false);
            delete raw;
            if (sen) sens.push_back(sen);
        }
        if (sens.empty()) break;

        std::vector<Result> results(sens.size());
        const int B = static_cast<int>(sens.size());

        #pragma omp parallel
        {
            Scorer local(model_);
            #pragma omp for schedule(dynamic, 4)
            for (int i = 0; i < B; i++) {
                const size_t T = sens[i]->Size();
                results[i].labels.resize(T);
                results[i].path_scores.resize(T);
                local.Viterbi(*sens[i], results[i].labels,
                              &results[i].score, &results[i].path_scores);
                results[i].valid = true;
            }
        }

        // Write outputs serially in order.
        for (int i = 0; i < B; i++) {
            out << "score=" << results[i].score << "\n";
            for (size_t t = 0; t < results[i].labels.size(); ++t) {
                out << processor->GetLabelStr(results[i].labels[t])
                    << " " << results[i].path_scores[t] << "\n";
            }
            out << "\n";
            delete sens[i];
        }
    }
}

} // namespace wati
