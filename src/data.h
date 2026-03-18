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

    void LoadFeatures(std::istream& file);
    void SaveFeatures(std::ostream& file) const;

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
};

} // namespace wati