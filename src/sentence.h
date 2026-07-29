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

// Compact Pos: 8 bytes (vs 64 before with 2 std::vectors).
struct Sentence {
    struct Pos {
        int32_t label = -1;
        uint16_t unigram_count = 0;
        uint16_t bigram_count = 0;
    };

    std::vector<Pos> pos;

    // obs storage: either owned in obs_buffer or external (mmap-backed).
    // If obs_external is non-null, it takes precedence and obs_buffer is unused.
    std::vector<int32_t> obs_buffer;
    const int32_t* obs_external = nullptr;

    uint32_t uni_stride = 0;
    uint32_t bi_stride = 0;

    size_t Size() const { return pos.size(); }

    inline const int32_t* obs_data() const {
        return obs_external ? obs_external : obs_buffer.data();
    }
    inline const int32_t* unigram_obs(uint32_t t) const {
        return obs_data() + t * (uni_stride + bi_stride);
    }
    inline const int32_t* bigram_obs(uint32_t t) const {
        return obs_data() + t * (uni_stride + bi_stride) + uni_stride;
    }
};

struct Dataset {
    std::vector<Sentence*> sens;
    uint32_t max_sentence_size = 0;

    // mmap-backed shared obs region (optional)
    void* obs_mmap = nullptr;
    size_t obs_mmap_size = 0;
    int obs_mmap_fd = -1;
#ifdef _WIN32
    // Windows fallback for LoadBinary: own the observation bytes in memory.
    std::vector<int32_t> obs_storage;
#endif

    ~Dataset();

    size_t Size() const { return sens.size(); }
};

} // namespace
