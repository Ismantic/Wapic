#include <iostream>
#include <fstream>

#include "option.h"
#include "model.h"
#include "data.h"
#include "optimize.h"
#include "score.h"

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
                                       option.L1, option.L2);
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
    }

    return 0;
}