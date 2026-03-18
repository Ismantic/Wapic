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

    for (uint32_t t = 0; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        for (int64_t y = 0; y < Y; ++y) {
            float_t sum = 0.0;
            for (uint32_t n = 0; n < pos.unigram_count; ++n) {
                const auto w = model_->GetUnigramWeights(pos.unigram_obs[n]);
                sum += w[y];
            }
            for (int64_t yp = 0; yp < Y; yp++) {
                psi_[t][yp][y] = sum;
            }
        }
    }
}

void Scorer::ComputeBigramScores(const Sentence& sen) {
    const size_t T = sen.Size();
    const int64_t Y = model_->LabelCount();

    for (uint32_t t = 1; t < T; ++t) {
        const Sentence::Pos& pos = sen.pos[t];
        for (int64_t yp = 0, d = 0; yp < Y; yp++) {
            for (int64_t y = 0; y < Y; y++, d++) {
                float_t sum = 0.0;
                for (uint32_t n = 0; n < pos.bigram_count; n++) {
                    const auto w = model_->GetBigramWeights(pos.bigram_obs[n]);
                    sum += w[d];
                }
                psi_[t][yp][y] += sum;
            }
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
    while (true) {
        RawStrs* raw = processor->ReadRawStrs(in);
        if (raw == nullptr) {
            break;
        }

        Sentence* sen = processor->RawToSentence(raw, false);
        const size_t T = sen->Size();
        std::vector<int64_t> labels(T);
        float_t score;
        std::vector<float_t> path_scores(T);
        Viterbi(*sen, labels, &score, &path_scores);
        out << "score=" << score << "\n";
        for (size_t t = 0; t < T; ++t) {
            auto a = labels[t];
            std::string s = processor->GetLabelStr(a);
            out << s << " " << path_scores[t] << "\n";
        }
        out << "\n";
        delete raw;
        delete sen;
    }
}

} // namespace wati