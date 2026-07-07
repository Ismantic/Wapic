#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

#include "option.h"
#include "model.h"
#include "data.h"
#include "optimize.h"
#include "preprocess.h"
#include "score.h"


int main(int argc, char* argv[]) {

    wati::Option option;
    std::string error_msg;

    if (!wati::OptionParser::Parse(argc, argv, option, error_msg)) {
        std::cerr << "Error: " << error_msg << "\n";
        return 1;
    }

    try {
        switch (option.run_mode) {
        case wati::RunMode::BUILD: {
            wati::DataProcessor processor;
            processor.LoadPatterns(option.pattern_file);
            std::ifstream in(option.input_file);
            if (!in) {
                std::cerr << "Cannot open input " << option.input_file << "\n";
                return 1;
            }
            processor.BuildBinary(in, option.output_file, option.nthread,
                                  option.min_count);
            break;
        }
        case wati::RunMode::CONVERT: {
            // In convert mode, the lone positional arg is the output (input via -m).
            std::string out_path = option.output_file.empty() ? option.input_file
                                                              : option.output_file;
            wati::Model model(std::make_unique<wati::DataProcessor>());
            std::cerr << "Loading " << option.model_file << " ..." << std::flush;
            model.Load(option.model_file);
            std::cerr << " done.\n";

            if (option.prune_threshold > 0.0) {
                auto& theta = model.GetWeights();
                int64_t zeroed = 0;
                for (auto& w : theta) {
                    if (std::abs(w) < option.prune_threshold) {
                        w = 0.0;
                        zeroed++;
                    }
                }
                std::cerr << "[prune-threshold] zeroed " << zeroed << "\n";
            }
            model.Save(out_path, option.save_binary, option.save_prune);
            std::cerr << "Saved to " << out_path << "\n";
            break;
        }
        case wati::RunMode::FIT: {
            wati::Model model(std::make_unique<wati::DataProcessor>());
            if (option.init_from.empty()) {
                model.LoadPatterns(option.pattern_file);
            } else {
                std::cerr << "Warm-start: loading " << option.init_from
                          << " ..." << std::flush;
                model.Load(option.init_from);
                model.LockFeatures();
                std::cerr << " done. (features locked, will continue from these weights)\n";
            }
            if (option.from_binary) {
                model.LoadDataBinary(option.input_file);
            } else {
                // LoadDataset dispatches on trie state: warm-start (locked) uses
                // the fully parallel path; from-scratch (unlocked) parallelizes
                // feature extraction and keeps trie inserts serial + in-order.
                model.LoadData(option.input_file, option.nthread);
            }
            if (option.min_count > 1) {
                model.PruneRareFeatures(option.min_count);
            }
            model.Sync();

            if (option.optimizer_type == wati::OptimizerType::SGD) {
                wati::SGDOptimizer s(&model, option.max_iterations,
                                     option.stop_window,
                                     option.stop_epsilon,
                                     option.GetOptimizerSpec<wati::SGD>()->learning_rate,
                                     option.GetOptimizerSpec<wati::SGD>()->decay_rate,
                                     option.L1);
                s.Optimize();
            } else {
                wati::LBFGSOptimizer s(&model,
                                       option.stop_window,
                                       option.stop_epsilon,
                                       option.max_iterations,
                                       option.objective_window,
                                       option.GetOptimizerSpec<wati::LBFGS>()->history_size,
                                       option.GetOptimizerSpec<wati::LBFGS>()->max_line_search,
                                       option.L1, option.L2,
                                       option.nthread);
                if (option.save_every > 0) {
                    s.SetCheckpoint(option.output_file, option.save_every);
                }
                s.Optimize();
            }

            // Optional: zero out small-magnitude weights before save (aggressive prune).
            if (option.prune_threshold > 0.0) {
                auto& theta = model.GetWeights();
                int64_t zeroed = 0;
                for (auto& w : theta) {
                    if (std::abs(w) < option.prune_threshold) {
                        w = 0.0;
                        zeroed++;
                    }
                }
                std::cerr << "[prune-threshold] zeroed " << zeroed
                          << " weights below " << option.prune_threshold << std::endl;
            }
            model.Save(option.output_file, option.save_binary, option.save_prune);
            break;
        }
        case wati::RunMode::LABEL: {
            std::ifstream input(option.input_file);
            if (!input) {
                throw std::runtime_error("Cannot open input: " + option.input_file);
            }

            wati::Model model(std::make_unique<wati::DataProcessor>());
            model.Load(option.model_file);
            // Inference-only: obs strings won't be needed (DAT handles lookups).
            model.FreeObservationStrings();

            wati::Scorer s(&model);

            std::ofstream output(option.output_file);
            if (!output) {
                throw std::runtime_error("Cannot open output: " + option.output_file);
            }

            s.LabelSentences(input, output);

            break;
        }
        case wati::RunMode::REPL: {
            wati::Model model(std::make_unique<wati::DataProcessor>());
            std::cerr << "Loading model..." << std::flush;
            model.Load(option.model_file);
            // Inference-only: obs strings won't be needed (DAT handles lookups).
            model.FreeObservationStrings();
            std::cerr << " done.\n";

            const wati::DataProcessor* processor = model.GetDataProcessor();
            wati::Scorer scorer(&model);

            std::cerr << "Type Chinese text, press Enter. Ctrl+D to quit.\n";
            std::string line;
            while (true) {
                std::cerr << ">>> " << std::flush;
                if (!std::getline(std::cin, line)) break;
                if (line.empty()) continue;
                if (line == "q" || line == "quit" || line == "exit") break;

                // Pre-segment: only Han runs go to the CRF; latin / digit /
                // punctuation runs become tokens directly. Mirrors the retag2
                // tokenization the model was trained on. Training paths (fit /
                // test) do not use this.
                std::vector<std::string> words;
                for (const auto& run : wati::PreSegment(line)) {
                    if (run.type == wati::RunType::Space) continue;  // boundary
                    if (run.type != wati::RunType::Han) {
                        words.push_back(run.text);
                        continue;
                    }

                    // Columnar CRF over this Han run: one char per line.
                    auto chars = wati::Utf8Chars(run.text);
                    if (chars.empty()) continue;
                    std::string buf;
                    for (auto& c : chars) { buf += c; buf += '\n'; }
                    buf += '\n';

                    std::istringstream iss(buf);
                    wati::RawStrs* raw = processor->ReadRawStrs(iss);
                    if (!raw) continue;
                    wati::Sentence* sen = processor->RawToSentence(raw, false);
                    if (!sen) { delete raw; continue; }

                    std::vector<int64_t> labels;
                    scorer.Viterbi(*sen, labels);

                    std::string cur;
                    for (size_t t = 0; t < chars.size() && t < labels.size(); t++) {
                        cur += chars[t];
                        std::string tag = processor->GetLabelStr(labels[t]);
                        if (tag == "E" || tag == "S") {
                            words.push_back(cur);
                            cur.clear();
                        }
                    }
                    if (!cur.empty()) words.push_back(cur);

                    delete raw;
                    delete sen;
                }

                // Join words with single spaces.
                std::string result;
                for (size_t k = 0; k < words.size(); k++) {
                    if (k) result += ' ';
                    result += words[k];
                }
                std::cout << result << "\n";
            }
            break;
        }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
