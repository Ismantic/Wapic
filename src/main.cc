#include <iostream>
#include <fstream>
#include <sstream>

#include "option.h"
#include "model.h"
#include "data.h"
#include "optimize.h"
#include "score.h"

// Decode a single UTF-8 character, return bytes consumed (1-4), or 0 on error
static int utf8_char_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

// Split a UTF-8 string into individual characters
static std::vector<std::string> utf8_chars(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while (i < s.size()) {
        int len = utf8_char_len(static_cast<unsigned char>(s[i]));
        chars.push_back(s.substr(i, len));
        i += len;
    }
    return chars;
}

int main(int argc, char* argv[]) {

    wati::Option option;
    std::string error_msg;

    if (!wati::OptionParser::Parse(argc, argv, option, error_msg)) {
        std::cerr << "Error: " << error_msg << "\n";
        return 1;
    }

    switch (option.run_mode) {
        case wati::RunMode::FIT: {
            wati::Model model(std::make_unique<wati::DataProcessor>());
            model.LoadPatterns(option.pattern_file);
            model.LoadData(option.input_file);
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
                s.Optimize();
            }

            model.Save(option.output_file);
            break;
        }
        case wati::RunMode::LABEL: {
            wati::Model model(std::make_unique<wati::DataProcessor>());
            model.Load(option.model_file);

            wati::Scorer s(&model);

            std::ifstream input(option.input_file);
            std::ofstream output(option.output_file);

            s.LabelSentences(input, output);

            break;
        }
        case wati::RunMode::REPL: {
            wati::Model model(std::make_unique<wati::DataProcessor>());
            std::cerr << "Loading model..." << std::flush;
            model.Load(option.model_file);
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

                // Split input into UTF-8 characters, build CRF input
                auto chars = utf8_chars(line);
                if (chars.empty()) continue;

                // Build columnar input: one char per line, blank line to end sentence
                std::string buf;
                for (auto& c : chars) {
                    buf += c;
                    buf += '\n';
                }
                buf += '\n';

                std::istringstream iss(buf);
                wati::RawStrs* raw = processor->ReadRawStrs(iss);
                if (!raw) continue;

                wati::Sentence* sen = processor->RawToSentence(raw, false);
                if (!sen) { delete raw; continue; }

                std::vector<int64_t> labels;
                scorer.Viterbi(*sen, labels);

                // Reconstruct segmented text from BMES tags
                std::string result;
                for (size_t t = 0; t < chars.size() && t < labels.size(); t++) {
                    std::string tag = processor->GetLabelStr(labels[t]);
                    result += chars[t];
                    if (tag == "E" || tag == "S") {
                        result += ' ';
                    }
                }
                // Trim trailing space
                if (!result.empty() && result.back() == ' ') {
                    result.pop_back();
                }

                std::cout << result << "\n";

                delete raw;
                delete sen;
            }
            break;
        }
    }

    return 0;
}