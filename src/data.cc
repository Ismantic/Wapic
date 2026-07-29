#include "data.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <unordered_map>
#ifndef _WIN32
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#endif

#include "misc.h"

namespace wati {

Dataset::~Dataset() {
    for (auto sen : sens) {
        delete sen;
    }
#ifndef _WIN32
    if (obs_mmap != nullptr) {
        munmap(obs_mmap, obs_mmap_size);
    }
    if (obs_mmap_fd >= 0) {
        close(obs_mmap_fd);
    }
#endif
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
    if (!is) {
        throw std::runtime_error("Cannot open pattern file: " + filename);
    }
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
                throw std::runtime_error("Invalid pattern: " + line);
        }

        patterns.push_back(pattern);
        token_count = std::max(token_count, pattern->TokenNum());
    }
    if (patterns.empty()) {
        throw std::runtime_error("Pattern file contains no patterns: " + filename);
    }
}

RawStrs* DataProcessor::ReadRawStrs(std::istream& file) const {
    if (!file) {
        return nullptr;
    }

    RawStrs* raw = new RawStrs();

    std::string source_line;
    while (std::getline(file, source_line)) {
        std::string line = TrimLine(source_line);
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

namespace {

// Route an observation id into the per-position obs slots by its kind
// ('u' unigram, 'b' bigram, '*' both).
inline void DispatchObs(char kind, int32_t id, int32_t*& uni, int32_t*& bi,
                        Sentence::Pos& pos) {
    switch (kind) {
        case 'u': *uni++ = id; pos.unigram_count++; break;
        case 'b': *bi++  = id; pos.bigram_count++;  break;
        case '*':
            *uni++ = id; *bi++ = id;
            pos.unigram_count++; pos.bigram_count++;
            break;
    }
}

}  // namespace

Sentence* DataProcessor::TokensToSentence(const TokenStrs* tos) const {
    Sentence* sen = new Sentence();
    sen->pos.resize(tos->Size());
    sen->uni_stride = unigram_count;
    sen->bi_stride = bigram_count;

    const uint32_t stride = unigram_count + bigram_count;
    sen->obs_buffer.assign(tos->Size() * stride, 0);
    int32_t* buf = sen->obs_buffer.data();

    std::string obs;  // reused across patterns/positions (see Pattern::Execute)
    for (uint32_t t = 0; t < tos->Size(); t++) {
        Sentence::Pos& pos = sen->pos[t];

        int32_t* unigram_start = buf + t*stride;
        int32_t* bigram_start = unigram_start + unigram_count;
        int32_t* unigram_current = unigram_start;
        int32_t* bigram_current = bigram_start;

        for (Pattern* pattern : patterns) {
            pattern->Execute(*tos, t, obs);
            int64_t i = observations->Insert(obs); // Lock when in test
            if (i == -1) continue;
            DispatchObs(obs[0], static_cast<int32_t>(i),
                        unigram_current, bigram_current, pos);
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


Dataset* DataProcessor::LoadDataset(std::istream& file, bool e, uint32_t nthread) {
    Dataset* data = new Dataset();

    if (nthread <= 1) {
        // Serial path (original).
        while (!file.eof()) {
            Sentence* sen = GetSentence(file, e);
            if (!sen) break;
            data->sens.push_back(sen);
            data->max_sentence_size = std::max(data->max_sentence_size,
                                               static_cast<uint32_t>(sen->Size()));
        }
        return data;
    }

    if (!observations->IsLocked()) {
        // From-scratch training: the tries are mutable, so Insert must stay
        // serial. Run two phases per batch (same pattern as BuildBinary):
        // parallel RawToTokens + Pattern::Execute, then in-order serial trie
        // inserts. Insertion order matches the serial path exactly, so obs
        // ids — and any model trained from them — are identical.
        LoadDatasetUnlocked(file, e, nthread, data);
        return data;
    }

    // Parallel pipelined path: a single producer thread reads RawStrs into a
    // double-buffer; the main thread runs OpenMP-parallel RawToSentence on the
    // other buffer. Append to dataset is serial but tiny.
    //
    // Safe only when label/obs tries are locked (warm-start); caller enforces.
    const size_t batch_size = 4096;
    struct Buffer {
        std::vector<RawStrs*> batch;
        std::vector<Sentence*> sens;
        bool eof = false;
    };
    Buffer buf_a, buf_b;
    buf_a.batch.reserve(batch_size);
    buf_b.batch.reserve(batch_size);

    auto fill_buffer = [&](Buffer& b) {
        b.batch.clear();
        b.eof = false;
        while (b.batch.size() < batch_size) {
            RawStrs* raw = ReadRawStrs(file);
            if (!raw) { b.eof = true; break; }
            b.batch.push_back(raw);
        }
    };

    // Prefill first buffer.
    fill_buffer(buf_a);

    Buffer* cur = &buf_a;
    Buffer* next = &buf_b;
    std::thread reader;  // background reader for next batch

    while (!cur->batch.empty()) {
        // Kick off async read for the NEXT batch while we process current.
        if (!cur->eof) {
            reader = std::thread([&]{ fill_buffer(*next); });
        }

        // Process current batch in parallel.
        const int B = static_cast<int>(cur->batch.size());
        cur->sens.assign(B, nullptr);

        #pragma omp parallel for schedule(static) num_threads(nthread)
        for (int i = 0; i < B; i++) {
            cur->sens[i] = RawToSentence(cur->batch[i], e);
        }

        // Serial append + delete raw (tiny work).
        for (int i = 0; i < B; i++) {
            if (cur->sens[i]) {
                data->sens.push_back(cur->sens[i]);
                data->max_sentence_size = std::max(data->max_sentence_size,
                                                   static_cast<uint32_t>(cur->sens[i]->Size()));
            }
            delete cur->batch[i];
            cur->batch[i] = nullptr;
        }

        // Wait for next batch read to finish; swap.
        if (reader.joinable()) reader.join();
        if (cur->eof) break;
        std::swap(cur, next);
    }

    return data;
}

// Shared batch driver. Reads sentences in batches of 512, tokenizes and runs
// Pattern::Execute in parallel, then hands each batch's PatternedSentences to
// `consume` serially and in input order — so trie insertion order (and thus
// observation ids) is identical to a fully serial pass.
void DataProcessor::ForEachPatternedBatch(
        std::istream& file, bool e, uint32_t nthread,
        const std::function<void(PatternedSentence&)>& consume) {
    const size_t batch_size = 512;
    std::vector<RawStrs*> batch;
    std::vector<PatternedSentence> results;
    batch.reserve(batch_size);
    results.reserve(batch_size);

    bool eof = false;
    while (!eof) {
        batch.clear();
        results.clear();
        while (batch.size() < batch_size && !eof) {
            RawStrs* raw = ReadRawStrs(file);
            if (!raw) { eof = true; break; }
            batch.push_back(raw);
        }
        if (batch.empty()) break;

        // Parallel phase: no shared mutable state.
        results.resize(batch.size());
        const int B = static_cast<int>(batch.size());
        #pragma omp parallel for schedule(static) num_threads(static_cast<int>(std::max(1u, nthread)))
        for (int i = 0; i < B; i++) {
            TokenStrs* tos = RawToTokens(batch[i], e);
            delete batch[i];
            batch[i] = nullptr;
            if (!tos || tos->Size() == 0) { delete tos; continue; }
            PatternedSentence& ps = results[i];
            ps.T = tos->Size();
            ps.obs.reserve(ps.T * patterns.size());
            if (!tos->labels.empty()) ps.labels.reserve(ps.T);
            std::string obs;  // reused across patterns/positions
            for (uint32_t t = 0; t < ps.T; t++) {
                for (Pattern* p : patterns) {
                    p->Execute(*tos, t, obs);
                    ps.obs.push_back(obs);
                }
                if (!tos->labels.empty()) ps.labels.push_back(tos->labels[t]);
            }
            delete tos;
        }

        // Serial phase, in input order.
        for (auto& ps : results) {
            if (ps.T != 0) consume(ps);
        }
    }
}

// Parse text once and write 3 sidecar files (obs.bin, meta.bin, trie.txt) so
// future training runs can mmap them in seconds instead of reparsing.
// With nthread>1, Pattern::Execute runs in parallel; Trie::Insert stays serial.
void DataProcessor::BuildBinary(std::istream& file, const std::string& prefix,
                                uint32_t nthread, uint32_t min_count) {
    const std::string obs_path  = prefix + ".obs.bin";
    const std::string meta_path = prefix + ".meta.bin";
    const std::string trie_path = prefix + ".trie.txt";

    // Pass 0 (only with min_count > 1): stream the corpus once counting how
    // often each observation string occurs, so the main pass can drop the
    // rare ones.
    std::unordered_map<std::string, uint32_t> obs_freq;
    if (min_count > 1) {
        uint64_t counted = 0;
        ForEachPatternedBatch(file, true, nthread, [&](PatternedSentence& ps) {
            for (auto& obs : ps.obs) obs_freq[std::move(obs)]++;
            counted += ps.T;
            if ((counted / 1000000) != ((counted - ps.T) / 1000000)) {
                std::cerr << "count: " << counted << " positions, "
                          << obs_freq.size() << " unique obs\r";
            }
        });
        std::cerr << "\nmin-count pass: " << obs_freq.size()
                  << " unique observations counted\n";
        file.clear();
        file.seekg(0);
    }

    std::ofstream obs_out(obs_path, std::ios::binary);
    std::ofstream meta_out(meta_path, std::ios::binary);
    if (!obs_out || !meta_out) {
        throw std::runtime_error("BuildBinary: cannot open output files for prefix " +
                                 prefix);
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

    ForEachPatternedBatch(file, true, nthread, [&](PatternedSentence& ps) {
        const uint32_t T = ps.T;
        std::vector<int32_t> sen_obs(T * stride, 0);
        std::vector<Sentence::Pos> sen_pos(T);

        size_t obs_idx = 0;
        for (uint32_t t = 0; t < T; t++) {
            Sentence::Pos& pos = sen_pos[t];
            int32_t* uni = sen_obs.data() + t * stride;
            int32_t* bi  = uni + unigram_count;
            for (size_t pi = 0; pi < patterns.size(); pi++) {
                const std::string& obs = ps.obs[obs_idx++];
                if (min_count > 1) {
                    auto it = obs_freq.find(obs);
                    if (it == obs_freq.end() || it->second < min_count) continue;
                }
                int64_t i = observations->Insert(obs);
                if (i == -1) continue;
                DispatchObs(obs[0], static_cast<int32_t>(i), uni, bi, pos);
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
    });

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

// From-scratch parallel load (mutable tries): parallel feature extraction,
// serial in-order trie inserts. Ids match the serial path exactly, so any
// model trained from this dataset is identical.
void DataProcessor::LoadDatasetUnlocked(std::istream& file, bool e,
                                        uint32_t nthread, Dataset* data) {
    const uint32_t stride = unigram_count + bigram_count;
    ForEachPatternedBatch(file, e, nthread, [&](PatternedSentence& ps) {
        const uint32_t T = ps.T;
        Sentence* sen = new Sentence();
        sen->pos.resize(T);
        sen->uni_stride = unigram_count;
        sen->bi_stride = bigram_count;
        sen->obs_buffer.assign(static_cast<size_t>(T) * stride, 0);
        int32_t* buf = sen->obs_buffer.data();

        size_t obs_idx = 0;
        for (uint32_t t = 0; t < T; t++) {
            Sentence::Pos& pos = sen->pos[t];
            int32_t* uni = buf + static_cast<size_t>(t) * stride;
            int32_t* bi  = uni + unigram_count;
            for (size_t pi = 0; pi < patterns.size(); pi++) {
                const std::string& obs = ps.obs[obs_idx++];
                int64_t id = observations->Insert(obs);
                if (id == -1) continue;
                DispatchObs(obs[0], static_cast<int32_t>(id), uni, bi, pos);
            }
            if (!ps.labels.empty()) {
                pos.label = static_cast<int32_t>(labels->Insert(ps.labels[t]));
            }
        }
        data->sens.push_back(sen);
        data->max_sentence_size = std::max(data->max_sentence_size, T);
    });
}

void DataProcessor::PruneRareObservations(Dataset* data, uint32_t min_count) {
    if (data->obs_mmap != nullptr) {
        throw std::runtime_error("--min-count cannot rewrite an mmap'd binary cache");
    }
    const int64_t O = static_cast<int64_t>(observations->Size());

    // Count occurrences (a '*' observation is counted from both lists; only
    // its total matters for the threshold).
    std::vector<uint32_t> cnt(O, 0);
    for (Sentence* sen : data->sens) {
        for (uint32_t t = 0; t < sen->Size(); t++) {
            const Sentence::Pos& pos = sen->pos[t];
            const int32_t* u = sen->unigram_obs(t);
            for (uint16_t n = 0; n < pos.unigram_count; n++) cnt[u[n]]++;
            const int32_t* b = sen->bigram_obs(t);
            for (uint16_t n = 0; n < pos.bigram_count; n++) cnt[b[n]]++;
        }
    }

    // Rebuild the trie with survivors in original insertion order, so the kept
    // ids stay dense and ordered; remap[old] = new id or -1.
    Trie* kept = new Trie();
    std::vector<int32_t> remap(O, -1);
    for (int64_t i = 0; i < O; i++) {
        if (cnt[i] >= min_count) {
            remap[i] = static_cast<int32_t>(kept->Insert(observations->GetValue(i)));
        }
    }
    const int64_t K = static_cast<int64_t>(kept->Size());
    delete observations;
    observations = kept;

    // Rewrite sentences in place: compact surviving ids, zero the tail.
    for (Sentence* sen : data->sens) {
        int32_t* buf = sen->obs_buffer.data();
        const uint32_t stride = sen->uni_stride + sen->bi_stride;
        for (uint32_t t = 0; t < sen->Size(); t++) {
            Sentence::Pos& pos = sen->pos[t];
            int32_t* u = buf + static_cast<size_t>(t) * stride;
            uint16_t w = 0;
            for (uint16_t n = 0; n < pos.unigram_count; n++) {
                int32_t nid = remap[u[n]];
                if (nid >= 0) u[w++] = nid;
            }
            for (uint16_t n = w; n < pos.unigram_count; n++) u[n] = 0;
            pos.unigram_count = w;

            int32_t* b = u + sen->uni_stride;
            w = 0;
            for (uint16_t n = 0; n < pos.bigram_count; n++) {
                int32_t nid = remap[b[n]];
                if (nid >= 0) b[w++] = nid;
            }
            for (uint16_t n = w; n < pos.bigram_count; n++) b[n] = 0;
            pos.bigram_count = w;
        }
    }

    std::cerr << "min-count " << min_count << ": kept " << K << "/" << O
              << " observations ("
              << (O ? static_cast<int>(100.0 * K / O) : 0) << "%)" << std::endl;
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

    // mmap obs on POSIX; Windows uses an owned in-memory buffer.
#ifdef _WIN32
    std::ifstream obs_in(obs_path, std::ios::binary | std::ios::ate);
    if (!obs_in) {
        std::cerr << "LoadBinary: cannot open " << obs_path << "\n";
        return nullptr;
    }
    const std::streamsize obs_size = obs_in.tellg();
    if (obs_size < 0 || obs_size % sizeof(int32_t) != 0) {
        std::cerr << "LoadBinary: invalid observation cache size\n";
        return nullptr;
    }
    obs_in.seekg(0);
    Dataset* data = new Dataset();
    data->obs_storage.resize(
        static_cast<size_t>(obs_size) / sizeof(int32_t));
    if (!obs_in.read(
            reinterpret_cast<char*>(data->obs_storage.data()), obs_size)) {
        std::cerr << "LoadBinary: cannot read " << obs_path << "\n";
        delete data;
        return nullptr;
    }
    data->obs_mmap = data->obs_storage.data();
    data->obs_mmap_size = static_cast<size_t>(obs_size);
    const int32_t* obs_base = data->obs_storage.data();
#else
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
#endif

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

    std::cerr << "LoadBinary: " << sentence_count << " sentences loaded from "
              << obs_path << " (" << (data->obs_mmap_size / (1ULL << 30))
              << " GB)" << std::endl;
    return data;
}

void DataProcessor::LoadFeatures(std::istream& file) {
    std::string line;
    if (!std::getline(file, line) || line.rfind("#Patterns#", 0) != 0) {
        throw std::runtime_error("Invalid model: missing patterns header");
    }

    size_t start = line.find("#Patterns#")+10;
    size_t end = line.find('#', start);
    if (end == std::string::npos) {
        throw std::runtime_error("Invalid model: malformed patterns header");
    }

    int pattern_count = std::stoll(line.substr(start, end-start));
    token_count = std::stoll(line.substr(end+1));
    if (pattern_count <= 0 || pattern_count > 100000 || token_count > 100000) {
        throw std::runtime_error("Invalid model: pattern metadata out of range");
    }
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
