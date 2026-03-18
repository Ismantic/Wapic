#include "model.h"


namespace wati {

void Model::Sync() {
    label_count_ = processor_->LabelCount();
    observation_count_ = processor_->ObservationCount();

    int64_t F = 0;
    int64_t Y = label_count_;
    int64_t O = observation_count_;

    kind_.resize(O);
    uoff_.resize(O);
    boff_.resize(O);

    for (int64_t o = 0; o < O; o++) {
        const std::string& obs = processor_->GetObservationStr(o);
        switch (obs[0]) {
            case 'u': kind_[o] = 1; break;
            case 'b': kind_[o] = 2; break;
            case '*': kind_[o] = 3; break;
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

void Model::Save(const std::string& filename) const {
    std::ofstream file(filename);
    processor_->SaveFeatures(file);

    int64_t nact = 0;
    for (int64_t f = 0; f < feature_count_; f++) {
        if (theta_[f] != 0.0) nact++;
    }
    file << "#Model#" << nact << std::endl;
    for (int64_t f = 0; f < feature_count_; f++) {
        if (theta_[f] != 0.0) {
            file << f << "=" << std::hexfloat << theta_[f] << "\n";
        }
    }
}

void Model::Load(const std::string& filename) {
    std::ifstream file(filename);
    processor_->LoadFeatures(file);
    Sync();

    int64_t nact = 0;
    std::string line;

    std::getline(file, line);
    size_t start = line.find("#Model#") + 7;  // 7 = strlen("#Model#")
    nact = std::stoll(line.substr(start));

    for (int64_t i = 0; i < nact; i++) {
        std::getline(file, line);
        size_t pos = line.find('=');
        int64_t f = std::stoll(line.substr(0, pos));
        float_t v = std::stod(line.substr(pos+1));
        theta_[f] = v;
    }
}
} // namespace wati