#pragma once

#include <vector>
#include <string>

#include <stdint.h>

namespace wati {

using float_t = double;

struct RawStrs {
    std::vector<std::string> strs;

    size_t Size() const { return strs.size(); }
};

struct TokenStrs {
    std::vector<std::string> labels;
    std::vector<std::vector<std::string>> tokens;

    size_t Size() const { return tokens.size(); }
};

struct Sentence {
    struct Pos {
        int64_t label = -1;
        uint32_t unigram_count = 0;
        uint32_t bigram_count = 0;
        std::vector<int64_t> unigram_obs;
        std::vector<int64_t> bigram_obs;
    };

    std::vector<Pos> pos;

    size_t Size() const { return pos.size(); }
};

struct Dataset {
    std::vector<Sentence*> sens;
    uint32_t max_sentence_size = 0;

    ~Dataset() {
        for (auto sen : sens) {
            delete sen;
        }
    }

    size_t Size() const { return sens.size(); }
};

} // namespace