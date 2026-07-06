#include "model.h"

#include "misc.h"

namespace wati {

void Model::Sync() {
    label_count_ = processor_->LabelCount();
    observation_count_ = processor_->ObservationCount();

    int64_t F = 0;
    int64_t Y = label_count_;
    int64_t O = observation_count_;
    if (Y <= 0 || O <= 0) {
        throw std::runtime_error("Model has no labels or observations");
    }

    kind_.resize(O);
    uoff_.resize(O);
    boff_.resize(O);

    for (int64_t o = 0; o < O; o++) {
        const std::string& obs = processor_->GetObservationStr(o);
        if (obs.empty()) {
            throw std::runtime_error("Model contains an empty observation");
        }
        switch (obs[0]) {
            case 'u': kind_[o] = 1; break;
            case 'b': kind_[o] = 2; break;
            case '*': kind_[o] = 3; break;
            default:
                throw std::runtime_error("Model contains an invalid observation");
        }
        if (kind_[o] & 1)
            uoff_[o] = F, F += Y;
        if (kind_[o] & 2)
            boff_[o] = F, F += Y*Y;  
    }
    feature_count_ = F;

    theta_.resize(F, 0.0);

    processor_->LockLabels();
    processor_->LockObservations();
}

void Model::Save(const std::string& filename, bool binary, bool prune) const {
    std::ofstream file(filename, binary ? (std::ios::out | std::ios::binary) : std::ios::out);
    if (!file) {
        throw std::runtime_error("Cannot open model output: " + filename);
    }

    const int64_t Y = label_count_;
    const int64_t O = observation_count_;

    // Determine alive observations if pruning.
    std::vector<bool> obs_alive;
    int64_t n_alive = O;
    if (prune) {
        obs_alive.assign(O, false);
        n_alive = 0;
        for (int64_t o = 0; o < O; o++) {
            bool alive = false;
            if (kind_[o] & 1) {
                for (int64_t y = 0; y < Y; y++)
                    if (theta_[uoff_[o] + y] != 0.0) { alive = true; break; }
            }
            if (!alive && (kind_[o] & 2)) {
                for (int64_t d = 0; d < Y*Y; d++)
                    if (theta_[boff_[o] + d] != 0.0) { alive = true; break; }
            }
            obs_alive[o] = alive;
            if (alive) n_alive++;
        }
    }

    // Save patterns + tries (with optional obs pruning).
    processor_->SaveFeatures(file, binary, prune ? &obs_alive : nullptr, n_alive);

    // Count active weights + compute write order.
    int64_t nact = 0;
    for (int64_t f = 0; f < feature_count_; f++) {
        if (theta_[f] != 0.0) nact++;
    }

    if (binary) {
        // Header
        file << "#ModelBin32#" << nact << "\n";
        int64_t prev = -1;

        if (prune) {
            // Walk obs in trie order, compute new feature offsets, write deltas.
            int64_t new_f = 0;
            for (int64_t o = 0; o < O; o++) {
                if (!obs_alive[o]) continue;
                if (kind_[o] & 1) {
                    for (int64_t y = 0; y < Y; y++) {
                        if (theta_[uoff_[o] + y] != 0.0) {
                            WriteVarUInt(file, static_cast<uint64_t>((new_f + y) - prev));
                            float v = static_cast<float>(theta_[uoff_[o] + y]);
                            file.write(reinterpret_cast<const char*>(&v), sizeof(float));
                            prev = new_f + y;
                        }
                    }
                    new_f += Y;
                }
                if (kind_[o] & 2) {
                    for (int64_t d = 0; d < Y*Y; d++) {
                        if (theta_[boff_[o] + d] != 0.0) {
                            WriteVarUInt(file, static_cast<uint64_t>((new_f + d) - prev));
                            float v = static_cast<float>(theta_[boff_[o] + d]);
                            file.write(reinterpret_cast<const char*>(&v), sizeof(float));
                            prev = new_f + d;
                        }
                    }
                    new_f += Y*Y;
                }
            }
        } else {
            // No pruning: just delta-encode in original feature index.
            for (int64_t f = 0; f < feature_count_; f++) {
                if (theta_[f] != 0.0) {
                    WriteVarUInt(file, static_cast<uint64_t>(f - prev));
                    float v = static_cast<float>(theta_[f]);
                    file.write(reinterpret_cast<const char*>(&v), sizeof(float));
                    prev = f;
                }
            }
        }
    } else {
        file << "#Model#" << nact << std::endl;
        for (int64_t f = 0; f < feature_count_; f++) {
            if (theta_[f] != 0.0) {
                file << f << "=" << std::hexfloat << theta_[f] << "\n";
            }
        }
    }
    file.flush();
    if (!file) {
        throw std::runtime_error("Failed while writing model: " + filename);
    }
}

void Model::BuildDeadMasks() {
    const int64_t Y = label_count_;
    const int64_t O = observation_count_;
    unigram_dead_.assign(O, 0);
    bigram_dead_.assign(O, 0);
    for (int64_t o = 0; o < O; o++) {
        if (kind_[o] & 1) {
            bool dead = true;
            const float_t* w = theta_.data() + uoff_[o];
            for (int64_t y = 0; y < Y; y++) {
                if (w[y] != 0.0) { dead = false; break; }
            }
            unigram_dead_[o] = dead ? 1 : 0;
        } else {
            unigram_dead_[o] = 1;
        }
        if (kind_[o] & 2) {
            bool dead = true;
            const float_t* w = theta_.data() + boff_[o];
            for (int64_t d = 0; d < Y * Y; d++) {
                if (w[d] != 0.0) { dead = false; break; }
            }
            bigram_dead_[o] = dead ? 1 : 0;
        } else {
            bigram_dead_[o] = 1;
        }
    }
}

void Model::Load(const std::string& filename) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open model: " + filename);
    }
    processor_->LoadFeatures(file);
    Sync();

    int64_t nact = 0;
    std::string line;

    std::getline(file, line);
    if (line.find("#ModelBin32#") == 0) {
        nact = std::stoll(line.substr(12));
        if (nact < 0 || nact > feature_count_) {
            throw std::runtime_error("Invalid model: active feature count out of range");
        }
        int64_t prev = -1;
        for (int64_t i = 0; i < nact; i++) {
            uint64_t delta = ReadVarUInt(file);
            int64_t f = prev + static_cast<int64_t>(delta);
            float v;
            if (f < 0 || f >= feature_count_ ||
                !file.read(reinterpret_cast<char*>(&v), sizeof(float))) {
                throw std::runtime_error("Invalid model: corrupt fp32 weight entry");
            }
            theta_[f] = static_cast<double>(v);
            prev = f;
        }
    } else if (line.find("#ModelBin#") == 0) {
        // legacy binary fp64
        nact = std::stoll(line.substr(10));
        if (nact < 0 || nact > feature_count_) {
            throw std::runtime_error("Invalid model: active feature count out of range");
        }
        int64_t prev = -1;
        for (int64_t i = 0; i < nact; i++) {
            uint64_t delta = ReadVarUInt(file);
            int64_t f = prev + static_cast<int64_t>(delta);
            double v;
            if (f < 0 || f >= feature_count_ ||
                !file.read(reinterpret_cast<char*>(&v), sizeof(double))) {
                throw std::runtime_error("Invalid model: corrupt fp64 weight entry");
            }
            theta_[f] = v;
            prev = f;
        }
    } else if (line.find("#Model#") == 0) {
        // legacy text format
        size_t start = 7;
        nact = std::stoll(line.substr(start));
        if (nact < 0 || nact > feature_count_) {
            throw std::runtime_error("Invalid model: active feature count out of range");
        }
        for (int64_t i = 0; i < nact; i++) {
            if (!std::getline(file, line)) {
                throw std::runtime_error("Invalid model: truncated text weights");
            }
            size_t pos = line.find('=');
            if (pos == std::string::npos) {
                throw std::runtime_error("Invalid model: malformed text weight");
            }
            int64_t f = std::stoll(line.substr(0, pos));
            if (f < 0 || f >= feature_count_) {
                throw std::runtime_error("Invalid model: weight index out of range");
            }
            float_t v = std::stod(line.substr(pos+1));
            theta_[f] = v;
        }
    } else {
        throw std::runtime_error("Invalid model: missing weight header");
    }

    BuildDeadMasks();
}
} // namespace wati
