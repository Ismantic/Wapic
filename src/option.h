#pragma once

#include <string>
#include <iostream>
#include <stdexcept>
#include <limits>
#include <memory>
#include <thread>

#include <stdint.h>

namespace wati {

using float_t = double;

enum class RunMode {
    FIT,
    LABEL,
    REPL
};

enum class OptimizerType {
    SGD,
    LBFGS
};

class OptimizerSpec {
public:
    virtual ~OptimizerSpec() = default;
    virtual bool Validate(std::string& error_msg) const = 0;
};

class SGD : public OptimizerSpec {
public:
    float_t learning_rate = 0.8; 
    float_t decay_rate = 0.85;

    // eta0, alpha

    bool Validate(std::string& error_msg) const override {
        if (learning_rate <= 0.0) {
            error_msg = "Learning rate must be positive";
            return false;
        }
        if (decay_rate <= 0.0 || decay_rate >= 1.0) {
            error_msg = "Decay rate must be in (0,1)";
            return false;
        }
        return true;
    }
};

class LBFGS : public OptimizerSpec {
public:
    uint32_t history_size = 5;    // --histsz
    uint32_t max_line_search = 40; // --maxls
    
    bool Validate(std::string& error_msg) const override {
        if (history_size == 0) {
            error_msg = "History size must be positive";
            return false;
        }
        if (max_line_search == 0) {
            error_msg = "Max line search iterations must be positive";
            return false;
        }
        return true;
    }
    
};

class Option {
public:
    RunMode run_mode;
    OptimizerType optimizer_type;

    std::string input_file;
    std::string output_file;
    std::string pattern_file;
    std::string model_file;

    uint32_t max_iterations;
    uint32_t nthread;

    float_t L1;
    float_t L2;

    uint32_t objective_window;
    uint32_t stop_window;
    float_t stop_epsilon;

private:
    std::unique_ptr<OptimizerSpec> optimizer_spec_;

public:
    Option() 
    : run_mode(RunMode::FIT)
    , optimizer_type(OptimizerType::LBFGS)
    , max_iterations(100)
    , nthread(std::max(1u, std::thread::hardware_concurrency() / 2))
    , L1(0.5)
    , L2(0.0001)
    , objective_window(5)
    , stop_window(5)
    , stop_epsilon(0.02)
    {
        SetOptimizer(OptimizerType::LBFGS);
    }

    explicit Option(OptimizerType opt_t) : Option() {
        SetOptimizer(opt_t);
    }

    Option(const Option&) = delete;
    Option& operator=(const Option&) = delete;
    Option(Option&&) = default;
    Option& operator=(Option&&) = default;

    template<typename T>
    T* GetOptimizerSpec() {
        return dynamic_cast<T*>(optimizer_spec_.get());
    }
    
    template<typename T>
    const T* GetOptimizerSpec() const {
        return dynamic_cast<const T*>(optimizer_spec_.get());
    }
            
    void SetOptimizer(OptimizerType type) {
        optimizer_type = type;
        switch (type) {
            case OptimizerType::SGD:
                optimizer_spec_ = std::make_unique<SGD>();
                break;
            case OptimizerType::LBFGS:
                optimizer_spec_ = std::make_unique<LBFGS>();
                break;
        }
    }

    bool Validate(std::string& error_msg) const {
        if (L1 < 0.0) {
            error_msg = "L1 penalty must be non-negative";
            return false;
        }
        if (L2 < 0.0) {
            error_msg = "L2 penalty must be non-negative";
            return false;
        }
        if (objective_window == 0) {
            error_msg = "Objective window must be positive";
            return false;
        }
        if (stop_window == 0) {
            error_msg = "Stop window must be positive";
            return false;
        }
        if (stop_epsilon <= 0.0) {
            error_msg = "Stop epsilon must be positive";
            return false;
        }
        
        switch (run_mode) {
            case RunMode::FIT:
                if (input_file.empty()) {
                    error_msg = "Training input file is required";
                    return false;
                }
                if (output_file.empty()) {
                    error_msg = "Model output file is required";
                    return false;
                }
                if (pattern_file.empty()) {
                    error_msg = "Pattern file is required for training";
                    return false;
                }
                if (!model_file.empty()) {
                    error_msg = "Training resume via --model is not implemented";
                    return false;
                }
                break;
                
            case RunMode::LABEL:
                if (input_file.empty()) {
                    error_msg = "Input file is required for labeling";
                    return false;
                }
                if (model_file.empty()) {
                    error_msg = "Model file is required for labeling";
                    return false;
                }
                if (output_file.empty()) {
                    error_msg = "Output file is required for labeling";
                    return false;
                }
                break;

            case RunMode::REPL:
                if (model_file.empty()) {
                    error_msg = "Model file is required for REPL";
                    return false;
                }
                break;
        }
        
        if (optimizer_spec_) {
            return optimizer_spec_->Validate(error_msg);
        }
        return true;
    }
};

class OptionParser {
public:
    static bool Parse(int argc, char* argv[], Option& option, std::string& error_msg) {
        if (argc < 2) {
            PrintHelp(argv[0]);
            error_msg = "No arguments specified";
            return false;
        }

        std::string first_arg = argv[1];
        if (first_arg == "-h" || first_arg == "--help") {
            PrintHelp(argv[0]);
            std::exit(0);
        }

        // Try parsing first arg as mode; if not a mode, default to REPL
        int opts_start;
        if (first_arg == "fit" || first_arg == "test") {
            ParseMode(first_arg, option.run_mode, error_msg);
            opts_start = 2;
        } else {
            option.run_mode = RunMode::REPL;
            opts_start = 1;
        }

        if (!ParseOptions(argc - opts_start, argv + opts_start, option, error_msg)) {
            return false;
        }

        return option.Validate(error_msg);
    }

    static void PrintHelp(const std::string& program_name) {
        std::cout << 
            "CRF Sequence Labeling Tool\n\n"
            "USAGE:\n"
            "    " << program_name << " <mode> [options] [input] [output]\n\n"
            
            "MODES:\n"
            "    fit         Train a CRF model\n"
            "    test        Label sequences using trained model\n"
            
            "GLOBAL OPTIONS:\n"
            "    -h, --help              Show this help message\n"
            
            "FIT MODE:\n"
            "    -a, --algo ALGORITHM    Optimizer: sgd-l1, l-bfgs (default: l-bfgs)\n"
            "    -p, --pattern FILE      Pattern file for feature extraction\n"
            "    -i, --maxiter INT       Maximum iterations (default: 100)\n"
            "    -1, --rho1 FLOAT        L1 penalty (default: 0.5)\n"
            "    -2, --rho2 FLOAT        L2 penalty (default: 0.0001)\n"
            "    -o, --objwin INT        Objective window (default: 5)\n"
            "    -w, --stopwin INT       Stop window (default: 5)\n"
            "    -e, --stopeps FLOAT     Stop epsilon (default: 0.02)\n"
            "    -t, --nthread INT       Number of threads (default: auto)\n\n"
            
            "SGD-L1 OPTIONS:\n"
            "    --eta0 FLOAT            Learning rate (default: 0.8)\n"
            "    --alpha FLOAT           Decay rate (default: 0.85)\n\n"
            
            "L-BFGS OPTIONS:\n"
            "    --histsz INT            History size (default: 5)\n"
            "    --maxls INT             Max line search (default: 40)\n\n"
            
            "TEST MODE:\n"
            "    -m, --model FILE        Model file to load\n"
            
            "EXAMPLES:\n"
            "    " << program_name << " fit -p patterns.txt -a l-bfgs train.txt model.wac\n"
            "    " << program_name << " test -m model.wac test.txt result.txt\n"
            "    " << program_name << " -m model.wac\n\n";
    }
private:
    static bool ParseMode(const std::string& mode_str, RunMode& mode, std::string& error_msg) {
        if (mode_str == "fit") {
            mode = RunMode::FIT;
            return true;
        } else if (mode_str == "test") {
            mode = RunMode::LABEL;
            return true;
        } else {
            error_msg = "Unknown mode: " + mode_str;
            return false;
        }
    }

    static bool ParseOptimizer(const std::string& opt_str, OptimizerType& type, std::string& error_msg) {
        if (opt_str == "sgd-l1" || opt_str == "sgd") {
            type = OptimizerType::SGD;
            return true;
        } else if (opt_str == "l-bfgs" || opt_str == "lbfgs") {
            type = OptimizerType::LBFGS;
            return true;
        } else {
            error_msg = "Unknown optimizer: " + opt_str;
            return false;
        }
    }

    static bool ParseOptions(int argc, char* argv[], Option& option, std::string& error_msg) {
        for (int i = 0; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg[0] != '-') {
                if (option.input_file.empty()) {
                    option.input_file = arg;
                } else if (option.output_file.empty()) {
                    option.output_file = arg;
                } else {
                    error_msg = "Too many input files: " + arg;
                    return false;
                }
                continue;
            }
            
            try {
                if (arg == "-a" || arg == "--algo") {
                    if (++i >= argc) {
                        error_msg = "Missing algorithm argument";
                        return false;
                    }
                    OptimizerType opt_t;
                    if (!ParseOptimizer(argv[i], opt_t, error_msg)) {
                        return false;
                    }
                    option.SetOptimizer(opt_t);
                }
                else if (arg == "-p" || arg == "--pattern") {
                    if (++i >= argc) {
                        error_msg = "Missing pattern file";
                        return false;
                    }
                    option.pattern_file = argv[i];
                }
                else if (arg == "-m" || arg == "--model") {
                    if (++i >= argc) {
                        error_msg = "Missing model file";
                        return false;
                    }
                    option.model_file = argv[i];
                }
                else if (arg == "-i" || arg == "--maxiter") {
                    if (++i >= argc) {
                        error_msg = "Missing max iterations";
                        return false;
                    }
                    option.max_iterations = std::stoul(argv[i]);
                }
                else if (arg == "-1" || arg == "--rho1") {
                    if (++i >= argc) {
                        error_msg = "Missing L1 penalty";
                        return false;
                    }
                    option.L1 = std::stod(argv[i]);
                }
                else if (arg == "-2" || arg == "--rho2") {
                    if (++i >= argc) {
                        error_msg = "Missing L2 penalty";
                        return false;
                    }
                    option.L2 = std::stod(argv[i]);
                }
                else if (arg == "-o" || arg == "--objwin") {
                    if (++i >= argc) {
                        error_msg = "Missing objective window";
                        return false;
                    }
                    option.objective_window = std::stoul(argv[i]);
                }
                else if (arg == "-w" || arg == "--stopwin") {
                    if (++i >= argc) {
                        error_msg = "Missing stop window";
                        return false;
                    }
                    option.stop_window = std::stoul(argv[i]);
                }
                else if (arg == "-e" || arg == "--stopeps") {
                    if (++i >= argc) {
                        error_msg = "Missing stop epsilon";
                        return false;
                    }
                    option.stop_epsilon = std::stod(argv[i]);
                }
                
                else if (arg == "--eta0") {
                    if (++i >= argc) {
                        error_msg = "Missing learning rate";
                        return false;
                    }
                    if (auto* sgd_config = option.GetOptimizerSpec<SGD>()) {
                        sgd_config->learning_rate = std::stod(argv[i]);
                    }
                }
                else if (arg == "--alpha") {
                    if (++i >= argc) {
                        error_msg = "Missing decay rate";
                        return false;
                    }
                    if (auto* sgd_config = option.GetOptimizerSpec<SGD>()) {
                        sgd_config->decay_rate = std::stod(argv[i]);
                    }
                }
                
                else if (arg == "-t" || arg == "--nthread") {
                    if (++i >= argc) {
                        error_msg = "Missing thread count";
                        return false;
                    }
                    option.nthread = std::stoul(argv[i]);
                    if (option.nthread == 0) option.nthread = 1;
                }

                else if (arg == "--histsz") {
                    if (++i >= argc) {
                        error_msg = "Missing history size";
                        return false;
                    }
                    if (auto* lbfgs_config = option.GetOptimizerSpec<LBFGS>()) {
                        lbfgs_config->history_size = std::stoul(argv[i]);
                    }
                }
                else if (arg == "--maxls") {
                    if (++i >= argc) {
                        error_msg = "Missing max line search";
                        return false;
                    }
                    if (auto* lbfgs_config = option.GetOptimizerSpec<LBFGS>()) {
                        lbfgs_config->max_line_search = std::stoul(argv[i]);
                    }
                }
                
                else {
                    error_msg = "Unknown option: " + arg;
                    return false;
                }
            } catch (const std::exception& e) {
                error_msg = "Invalid value for " + arg + ": " + e.what();
                return false;
            }
        }
        
        return true;
    }    
};

} // namespace wati
