#include "data.h"

#include <iostream>
#include <fstream>
#include <algorithm>

#include "misc.h"

namespace wati {

DataProcessor::DataProcessor() : token_count(0), unigram_count(0), bigram_count(0) {
    labels = new Trie();
    observations = new Trie();
}

DataProcessor::~DataProcessor() {
    for (Pattern* p : patterns) {
        delete p;
    }
    delete labels;
    delete observations;
}

void DataProcessor::LoadPatterns(const std::string &filename) {
    std::ifstream is(filename);
    std::string line;
    while (std::getline(is, line)) {
        // Remove comments
        auto comment_pos = line.find('#');
        if (comment_pos != std::string::npos) {
            line.erase(comment_pos);
        }

        line = TrimLine(line);
        if (line.empty()) {
            continue;
        }

        line[0] = std::tolower(line[0]);

        Pattern* pattern = new Pattern(line);

        switch(line[0]) {
            case 'u': unigram_count++; break;
            case 'b': bigram_count++; break;
            case '*':
                unigram_count++;
                bigram_count++;
                break;
            default:
                delete pattern;
                return;
        }

        patterns.push_back(pattern);
        token_count = std::max(token_count, pattern->TokenNum());
    }
}

RawStrs* DataProcessor::ReadRawStrs(std::istream& file) const {
    if (file.eof()) {
        return nullptr;
    }

    RawStrs* raw = new RawStrs();

    while (!file.eof()) {
        std::string line = GetLine(file);
        if (line.empty()) {
            if (raw->strs.empty()) {
                continue;
            }
            break;
        }
        raw->strs.push_back(line);
    }

    if (raw->strs.empty()) {
        delete raw;
        return nullptr;
    }
    return raw;
}

TokenStrs* DataProcessor::RawToTokens(const RawStrs* raw, bool e) const {
    TokenStrs* tos = new TokenStrs();
    tos->tokens.resize(raw->Size());
    if (e) {
        tos->labels.reserve(raw->Size());
    }

    for (uint32_t t = 0; t < raw->Size(); t++) {
        std::string line = raw->strs[t];
        std::vector<std::string> tokens = SplitLine(line);

        if (e && !tokens.empty()) {
            tos->labels.push_back(tokens.back());
            tokens.pop_back();
        }

        tos->tokens[t] = std::move(tokens);
    }
    return tos;
}

Sentence* DataProcessor::TokensToSentence(const TokenStrs* tos) const {
    Sentence* sen = new Sentence();
    sen->pos.resize(tos->Size());

    std::vector<int64_t> raw;
    raw.resize(tos->Size() * (unigram_count + bigram_count));
    int64_t* raw_data = raw.data();

    for (uint32_t t = 0; t < tos->Size(); t++) {
        Sentence::Pos& pos = sen->pos[t];

        int64_t* unigram_start = raw_data + t*(unigram_count + bigram_count);
        int64_t* bigram_start = unigram_start + unigram_count;
        int64_t* unigram_current = unigram_start;
        int64_t* bigram_current = bigram_start;

        for (Pattern* pattern : patterns) {
            std::string obs = pattern->Execute(*tos, t);
            int64_t i = observations->Insert(obs); // Lock when in test

            if (i == -1) continue;

            switch(obs[0]) {
                case 'u': {
                    *unigram_current++ = i;
                    pos.unigram_count++;
                    break;
                }
                case 'b': {
                    *bigram_current++ = i;
                    pos.bigram_count++;
                    break;
                }
                case '*': {
                    *unigram_current++ = i;
                    *bigram_current++ = i;
                    pos.unigram_count++;
                    pos.bigram_count++;
                    break;
                }
            }
        }
        pos.unigram_obs.assign(unigram_start, unigram_start + pos.unigram_count);
        pos.bigram_obs.assign(bigram_start, bigram_start + pos.bigram_count);

        if (!tos->labels.empty()) {
            pos.label = labels->Insert(tos->labels[t]);
        }
    }

    return sen;
}

Sentence* DataProcessor::RawToSentence(const RawStrs* raw, bool e) const {
    TokenStrs* tos = RawToTokens(raw, e);
    if (!tos) return nullptr;

    Sentence* sen = TokensToSentence(tos);
    
    delete tos;
    return sen;
}

Sentence* DataProcessor::GetSentence(std::istream& file, bool e) const {
    RawStrs* raw = ReadRawStrs(file);
    if (!raw) return nullptr;

    Sentence* sen = RawToSentence(raw, e);
    delete raw;
    return sen;
}


Dataset* DataProcessor::LoadDataset(std::istream& file, bool e) {
    Dataset* data = new Dataset();

    while (!file.eof()) {
        Sentence* sen = GetSentence(file, e);
        if (!sen) break;

        data->sens.push_back(sen);
        data->max_sentence_size = std::max(data->max_sentence_size, 
                                           static_cast<uint32_t>(sen->Size()));
    }

    return data;
}

void DataProcessor::LoadFeatures(std::istream& file) {
    std::string line;
    std::getline(file, line);

    size_t start = line.find("#Patterns#")+10;
    size_t end = line.find('#', start);

    int pattern_count = std::stoll(line.substr(start, end-start));
    token_count = std::stoll(line.substr(end+1));
    unigram_count = bigram_count = 0;
    if (pattern_count > 0) {
        patterns.clear();
        patterns.reserve(pattern_count);
        for (int p = 0; p < pattern_count; p++) {
            std::string src = ReadStr(file);
            patterns.push_back(new Pattern(src));

            switch(std::tolower(src[0])) {
                case 'u': unigram_count++; break;
                case 'b': bigram_count++; break;
                case '*': unigram_count++; bigram_count++; break;
            }
        }
    }

    labels->Load(file);
    observations->Load(file);
}

void DataProcessor::SaveFeatures(std::ostream& file) const {
    file << "#Patterns#" << patterns.size() << "#" << token_count << "\n";
    for (uint32_t p = 0; p < patterns.size(); p++) {
        WriteStr(file, patterns[p]->GetSource());
    }

    labels->Save(file);
    observations->Save(file);
}

} // namespace wati