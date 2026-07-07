#pragma once

#include <functional>

#include "sentence.h"
#include "pattern.h"
#include "trie.h"

namespace wati {

class DataProcessor {
private:
    std::vector<Pattern*> patterns;
    Trie* labels;
    Trie* observations;
    uint32_t token_count;
    uint32_t unigram_count;
    uint32_t bigram_count;

    // Per-sentence intermediate from the parallel feature-extraction phase.
    struct PatternedSentence {
        uint32_t T = 0;
        std::vector<std::string> obs;     // T * pattern_count, ordered (t, pattern)
        std::vector<std::string> labels;  // T (empty when e == false)
    };

    TokenStrs* RawToTokens(const RawStrs* raw, bool e) const;
    Sentence* TokensToSentence(const TokenStrs* tos) const;
    Sentence* GetSentence(std::istream& file, bool e) const;

    // Shared batch driver: read sentences in batches, run RawToTokens +
    // Pattern::Execute in parallel, then hand each batch's results to
    // `consume` serially and in input order. Used by the from-scratch loader
    // and both BuildBinary passes.
    void ForEachPatternedBatch(std::istream& file, bool e, uint32_t nthread,
                               const std::function<void(PatternedSentence&)>& consume);

    // From-scratch parallel load: parallel feature extraction, serial in-order
    // trie inserts. Produces the same dataset/ids as the serial path.
    void LoadDatasetUnlocked(std::istream& file, bool e, uint32_t nthread,
                             Dataset* data);

public:
    DataProcessor();
    ~DataProcessor();

    RawStrs* ReadRawStrs(std::istream& file) const;
    Sentence* RawToSentence(const RawStrs* raw, bool e) const;

    void LoadPatterns(const std::string& filename);
    // nthread > 1 parallelizes per-sentence work (RawToTokens + Pattern::Execute
    // + trie lookup). Dispatches on trie state: locked tries (warm-start) use the
    // fully parallel pipelined path; unlocked (from-scratch) use parallel feature
    // extraction with serial in-order inserts. nthread <= 1 is the serial path.
    Dataset* LoadDataset(std::istream& file, bool e, uint32_t nthread = 1);

    // Binary cache pipeline: parse text once, write 3 files (<prefix>.obs.bin /
    // <prefix>.meta.bin / <prefix>.trie.txt). Later runs mmap them for fast load.
    // nthread > 1 enables parallel pattern execution; trie insert stays serial.
    // min_count > 1 makes it two-pass: count observation frequencies first,
    // then drop observations seen fewer than min_count times from the cache —
    // the way to apply --min-count to trainings that use --from-bin.
    void BuildBinary(std::istream& file, const std::string& prefix,
                     uint32_t nthread = 1, uint32_t min_count = 1);
    Dataset* LoadBinary(const std::string& prefix);

    // Drop observations occurring fewer than min_count times in `data`:
    // rebuilds the observation trie with the survivors (original order kept)
    // and rewrites every sentence's obs ids in place. Must run before Sync().
    // Not usable with mmap-backed datasets (read-only obs).
    void PruneRareObservations(Dataset* data, uint32_t min_count);

    void LoadFeatures(std::istream& file);
    // If obs_alive != nullptr, only writes observations marked true (n_alive count).
    void SaveFeatures(std::ostream& file, bool binary = false,
                      const std::vector<bool>* obs_alive = nullptr,
                      int64_t n_alive_count = 0) const;

    size_t LabelCount() const { return labels->Size(); } 
    size_t ObservationCount() const { return observations->Size(); }
    uint32_t UnigramCount() const { return unigram_count; }
    uint32_t BigramCount() const { return bigram_count; }

    std::string GetLabelStr(int64_t i) const {
        return labels->GetValue(i);
    }
    std::string GetObservationStr(int64_t i) const {
        return observations->GetValue(i);
    }

    void LockLabels() {
        labels->SetLock(true);
    }
    void LockObservations() {
        observations->SetLock(true);
    }

    // Inference-only: drop obs trie's value strings (saves ~hundreds of MB on
    // large models). Lookups still work via DAT; GetObservationStr(i) becomes
    // empty. Label strings are kept (needed to emit BMES tags).
    void FreeObservationStrings() {
        observations->FreeValueStrings();
    }
};

} // namespace wati