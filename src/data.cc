#include "data.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <deque>
#include <future>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "misc.h"

namespace wati {

Dataset::~Dataset() {
    for (auto sen : sens) {
        delete sen;
    }
    if (obs_mmap != nullptr) {
        munmap(obs_mmap, obs_mmap_size);
    }
    if (obs_mmap_fd >= 0) {
        close(obs_mmap_fd);
    }
}

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
    sen->uni_stride = unigram_count;
    sen->bi_stride = bigram_count;

    const uint32_t stride = unigram_count + bigram_count;
    sen->obs_buffer.assign(tos->Size() * stride, 0);
    int32_t* buf = sen->obs_buffer.data();

    for (uint32_t t = 0; t < tos->Size(); t++) {
        Sentence::Pos& pos = sen->pos[t];

        int32_t* unigram_start = buf + t*stride;
        int32_t* bigram_start = unigram_start + unigram_count;
        int32_t* unigram_current = unigram_start;
        int32_t* bigram_current = bigram_start;

        for (Pattern* pattern : patterns) {
            std::string obs = pattern->Execute(*tos, t);
            int64_t i = observations->Insert(obs); // Lock when in test

            if (i == -1) continue;

            switch(obs[0]) {
                case 'u': {
                    *unigram_current++ = static_cast<int32_t>(i);
                    pos.unigram_count++;
                    break;
                }
                case 'b': {
                    *bigram_current++ = static_cast<int32_t>(i);
                    pos.bigram_count++;
                    break;
                }
                case '*': {
                    *unigram_current++ = static_cast<int32_t>(i);
                    *bigram_current++ = static_cast<int32_t>(i);
                    pos.unigram_count++;
                    pos.bigram_count++;
                    break;
                }
            }
        }

        if (!tos->labels.empty()) {
            pos.label = static_cast<int32_t>(labels->Insert(tos->labels[t]));
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

namespace {

// Per-sentence intermediate result from worker threads.
struct PatternedSentence {
    uint32_t T = 0;
    std::vector<std::string> obs;     // size T * pattern_count, ordered by (t, pattern_idx)
    std::vector<std::string> labels;  // size T (may be empty if no labels)
};

}  // namespace

// Parse text once and write 3 sidecar files (obs.bin, meta.bin, trie.txt) so
// future training runs can mmap them in seconds instead of reparsing.
// With nthread>1, Pattern::Execute runs in parallel; Trie::Insert stays serial.
void DataProcessor::BuildBinary(std::istream& file, const std::string& prefix,
                                uint32_t nthread) {
    const std::string obs_path  = prefix + ".obs.bin";
    const std::string meta_path = prefix + ".meta.bin";
    const std::string trie_path = prefix + ".trie.txt";

    std::ofstream obs_out(obs_path, std::ios::binary);
    std::ofstream meta_out(meta_path, std::ios::binary);
    if (!obs_out || !meta_out) {
        std::cerr << "BuildBinary: cannot open output files\n";
        return;
    }

    // Meta header (rewritten at end with final sentence_count)
    uint64_t sentence_count = 0;
    uint32_t us = unigram_count;
    uint32_t bs = bigram_count;
    meta_out.write(reinterpret_cast<const char*>(&sentence_count), 8);
    meta_out.write(reinterpret_cast<const char*>(&us), 4);
    meta_out.write(reinterpret_cast<const char*>(&bs), 4);

    const uint32_t stride = unigram_count + bigram_count;
    uint64_t obs_pos_count = 0;  // running obs offset (in int32 units)

    uint64_t parsed_lines = 0;
    auto t_start = std::chrono::steady_clock::now();


    auto consume_one = [&](PatternedSentence&& ps) {
        if (ps.T == 0) return;
        const uint32_t T = ps.T;
        std::vector<int32_t> sen_obs(T * stride, 0);
        std::vector<Sentence::Pos> sen_pos(T);

        size_t obs_idx = 0;
        for (uint32_t t = 0; t < T; t++) {
            Sentence::Pos& pos = sen_pos[t];
            int32_t* unigram_start = sen_obs.data() + t * stride;
            int32_t* bigram_start  = unigram_start + unigram_count;
            int32_t* unigram_current = unigram_start;
            int32_t* bigram_current  = bigram_start;

            for (size_t pi = 0; pi < patterns.size(); pi++) {
                const std::string& obs = ps.obs[obs_idx++];
                int64_t i = observations->Insert(obs);
                if (i == -1) continue;
                switch (obs[0]) {
                    case 'u':
                        *unigram_current++ = static_cast<int32_t>(i);
                        pos.unigram_count++;
                        break;
                    case 'b':
                        *bigram_current++ = static_cast<int32_t>(i);
                        pos.bigram_count++;
                        break;
                    case '*':
                        *unigram_current++ = static_cast<int32_t>(i);
                        *bigram_current++ = static_cast<int32_t>(i);
                        pos.unigram_count++;
                        pos.bigram_count++;
                        break;
                }
            }
            if (!ps.labels.empty()) {
                pos.label = static_cast<int32_t>(labels->Insert(ps.labels[t]));
            }
        }

        uint32_t pos_count = T;
        uint64_t obs_off = obs_pos_count;
        meta_out.write(reinterpret_cast<const char*>(&pos_count), 4);
        meta_out.write(reinterpret_cast<const char*>(&obs_off), 8);
        meta_out.write(reinterpret_cast<const char*>(sen_pos.data()),
                       static_cast<std::streamsize>(pos_count * sizeof(Sentence::Pos)));
        obs_out.write(reinterpret_cast<const char*>(sen_obs.data()),
                      static_cast<std::streamsize>(sen_obs.size() * sizeof(int32_t)));
        obs_pos_count += sen_obs.size();
        sentence_count++;

        if ((++parsed_lines % 100000) == 0) {
            auto now = std::chrono::steady_clock::now();
            double dt = std::chrono::duration<double>(now - t_start).count();
            std::cerr << "build: " << parsed_lines << " sentences  ("
                      << static_cast<int64_t>(parsed_lines / std::max(dt, 1e-9))
                      << " sent/s)" << std::endl;
        }
    };

    // Main loop: read raw sentences only (no tokenize, no parse), then run
    // RawToTokens + Pattern::Execute in parallel via OpenMP, then serially
    // Trie-insert + write the results. Main does minimal serial work.
    const size_t batch_size = 512;
    std::vector<RawStrs*> batch;
    batch.reserve(batch_size);
    std::vector<PatternedSentence> results;
    results.reserve(batch_size);

    bool eof = false;
    while (!eof) {
        // Clean any leftover batch entries from prev iteration (defensive)
        for (auto* r : batch) delete r;
        batch.clear();
        results.clear();
        while (batch.size() < batch_size && !eof) {
            RawStrs* raw = ReadRawStrs(file);
            if (!raw) { eof = true; break; }
            batch.push_back(raw);
        }
        if (batch.empty()) break;

        results.resize(batch.size());
        const int B = static_cast<int>(batch.size());
        const int nt = std::max(1u, nthread);
        #pragma omp parallel for schedule(static) num_threads(nt)
        for (int i = 0; i < B; i++) {
            RawStrs* raw = batch[i];
            TokenStrs* tos = RawToTokens(raw, true);
            delete raw;
            batch[i] = nullptr;
            if (!tos || tos->Size() == 0) {
                delete tos;
                continue;
            }
            PatternedSentence& ps = results[i];
            ps.T = tos->Size();
            ps.obs.reserve(ps.T * patterns.size());
            if (!tos->labels.empty()) ps.labels.reserve(ps.T);
            for (uint32_t t = 0; t < ps.T; t++) {
                for (Pattern* p : patterns) {
                    ps.obs.push_back(p->Execute(*tos, t));
                }
                if (!tos->labels.empty()) ps.labels.push_back(tos->labels[t]);
            }
            delete tos;
        }

        for (auto& ps : results) consume_one(std::move(ps));
    }

    obs_out.close();

    meta_out.seekp(0);
    meta_out.write(reinterpret_cast<const char*>(&sentence_count), 8);
    meta_out.close();

    std::ofstream trie_out(trie_path);
    labels->Save(trie_out);
    observations->Save(trie_out);
    trie_out.close();

    std::cerr << "BuildBinary: " << sentence_count << " sentences, "
              << obs_pos_count << " obs ints (" << (obs_pos_count * 4.0 / (1<<30))
              << " GB)" << std::endl;
}

Dataset* DataProcessor::LoadBinary(const std::string& prefix) {
    const std::string obs_path  = prefix + ".obs.bin";
    const std::string meta_path = prefix + ".meta.bin";
    const std::string trie_path = prefix + ".trie.txt";

    // Load tries first (small, in RAM)
    std::ifstream trie_in(trie_path);
    if (!trie_in) {
        std::cerr << "LoadBinary: cannot open " << trie_path << "\n";
        return nullptr;
    }
    labels->Load(trie_in);
    observations->Load(trie_in);
    trie_in.close();

    // mmap obs
    int fd = open(obs_path.c_str(), O_RDONLY);
    if (fd < 0) {
        std::cerr << "LoadBinary: cannot open " << obs_path << "\n";
        return nullptr;
    }
    struct stat sb;
    if (fstat(fd, &sb) != 0) {
        std::cerr << "LoadBinary: fstat failed\n";
        close(fd);
        return nullptr;
    }
    void* mmap_base = mmap(nullptr, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_base == MAP_FAILED) {
        std::cerr << "LoadBinary: mmap failed\n";
        close(fd);
        return nullptr;
    }
    madvise(mmap_base, sb.st_size, MADV_SEQUENTIAL);

    Dataset* data = new Dataset();
    data->obs_mmap = mmap_base;
    data->obs_mmap_size = sb.st_size;
    data->obs_mmap_fd = fd;
    const int32_t* obs_base = reinterpret_cast<const int32_t*>(mmap_base);

    // Read meta
    std::ifstream meta_in(meta_path, std::ios::binary);
    if (!meta_in) {
        std::cerr << "LoadBinary: cannot open " << meta_path << "\n";
        delete data;
        return nullptr;
    }
    uint64_t sentence_count;
    uint32_t us, bs;
    meta_in.read(reinterpret_cast<char*>(&sentence_count), 8);
    meta_in.read(reinterpret_cast<char*>(&us), 4);
    meta_in.read(reinterpret_cast<char*>(&bs), 4);
    unigram_count = us;
    bigram_count  = bs;

    data->sens.reserve(sentence_count);
    for (uint64_t s = 0; s < sentence_count; s++) {
        uint32_t pos_count;
        uint64_t obs_off;
        meta_in.read(reinterpret_cast<char*>(&pos_count), 4);
        meta_in.read(reinterpret_cast<char*>(&obs_off), 8);

        Sentence* sen = new Sentence();
        sen->pos.resize(pos_count);
        meta_in.read(reinterpret_cast<char*>(sen->pos.data()),
                     static_cast<std::streamsize>(pos_count * sizeof(Sentence::Pos)));
        sen->obs_external = obs_base + obs_off;
        sen->uni_stride = us;
        sen->bi_stride  = bs;
        data->sens.push_back(sen);
        data->max_sentence_size = std::max(data->max_sentence_size, pos_count);
    }

    std::cerr << "LoadBinary: " << sentence_count << " sentences mmap'd from "
              << obs_path << " (" << (sb.st_size / (1<<30)) << " GB)" << std::endl;
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

    labels->LoadAuto(file);
    observations->LoadAuto(file);
}

void DataProcessor::SaveFeatures(std::ostream& file, bool binary,
                                 const std::vector<bool>* obs_alive,
                                 int64_t n_alive_count) const {
    file << "#Patterns#" << patterns.size() << "#" << token_count << "\n";
    for (uint32_t p = 0; p < patterns.size(); p++) {
        WriteStr(file, patterns[p]->GetSource());
    }

    if (binary) {
        labels->SaveBin(file);
    } else {
        labels->Save(file);
    }

    if (obs_alive) {
        // Write only alive observations, preserving original order.
        const int64_t O = static_cast<int64_t>(observations->Size());
        if (binary) {
            file << "#TrieBin#" << n_alive_count << "\n";
            for (int64_t o = 0; o < O; o++) {
                if ((*obs_alive)[o]) WriteStrBin(file, observations->GetValue(o));
            }
        } else {
            file << "#Trie#" << n_alive_count << "\n";
            for (int64_t o = 0; o < O; o++) {
                if ((*obs_alive)[o]) WriteStr(file, observations->GetValue(o));
            }
        }
    } else {
        if (binary) observations->SaveBin(file);
        else observations->Save(file);
    }
}

} // namespace wati