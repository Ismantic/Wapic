#pragma once 

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

    TokenStrs* RawToTokens(const RawStrs* raw, bool e) const;
    Sentence* TokensToSentence(const TokenStrs* tos) const;
    Sentence* GetSentence(std::istream& file, bool e) const;

public:
    DataProcessor();
    ~DataProcessor();

    RawStrs* ReadRawStrs(std::istream& file) const;
    Sentence* RawToSentence(const RawStrs* raw, bool e) const;

    void LoadPatterns(const std::string& filename);
    Dataset* LoadDataset(std::istream& file, bool e);

    // Binary cache pipeline: parse text once, write 3 files (<prefix>.obs.bin /
    // <prefix>.meta.bin / <prefix>.trie.txt). Later runs mmap them for fast load.
    // nthread > 1 enables parallel pattern execution; trie insert stays serial.
    void BuildBinary(std::istream& file, const std::string& prefix,
                     uint32_t nthread = 1);
    Dataset* LoadBinary(const std::string& prefix);

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
    void UnlockLabels() {
        labels->SetLock(false);
    }
    void UnlockObservations() {
        observations->SetLock(false);
    }

    // Inference-only: drop obs trie's value strings (saves ~hundreds of MB on
    // large models). Lookups still work via DAT; GetObservationStr(i) becomes
    // empty. Label strings are kept (needed to emit BMES tags).
    void FreeObservationStrings() {
        observations->FreeValueStrings();
    }
};

} // namespace wati