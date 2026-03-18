#pragma once

#include <vector>
#include <iomanip>

#include <stdint.h>

#include "sentence.h"
#include "model.h"
#include "state.h"

namespace wati {

class Scorer {
private:
    const Model* model_;
    std::vector<std::vector<std::vector<float_t>>> psi_; // [t][y'][y]

    struct BackData {
        std::vector<std::vector<int64_t>> back;
        std::vector<float_t> curr;
        std::vector<float_t> prev;

        BackData(size_t T, int64_t Y) 
            : back(T, std::vector<int64_t>(Y))
            , curr(Y)
            , prev(Y) {}
    };

    void ResizePsi(size_t T, int64_t Y);

    void ComputeUnigramScores(const Sentence& sen);
    void ComputeBigramScores(const Sentence& sen);
    bool ComputeScore(const Sentence& sen);

public:
    explicit Scorer(const Model* model) : model_(model) {}

    void Viterbi(const Sentence& sen, 
                 std::vector<int64_t>& labels, 
                 float_t* score = nullptr, 
                 std::vector<float_t>* path_scores = nullptr);
    
    void LabelSentences(std::istream& in, std::ostream& out);

};

} // namespace wati