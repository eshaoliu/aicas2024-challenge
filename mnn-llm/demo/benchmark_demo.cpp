// #include "llm.hpp"
// #include <fstream>
// #include <stdlib.h>

#include "llm.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>
#include <stdlib.h>

void benchmark(Llm* llm, std::string prompt_file, int max_decode_len, int num_tests) {
    std::cout << "prompt file is " << prompt_file << std::endl;
    std::ifstream prompt_fs(prompt_file);
    std::vector<std::string> prompts;
    std::string prompt;
    while (std::getline(prompt_fs, prompt)) {
        // prompt start with '#' will be ignored
        if (prompt.substr(0, 1) == "#") {
            continue;
        }
        prompts.push_back(prompt);
    }
    int prompt_len = 0;
    int decode_len = 0;
    int64_t prefill_time = 0;
    int64_t decode_time = 0;
    llm->warmup();
    for (int i = 0; i < num_tests; i++) {
        for (const auto& prompt : prompts) {
            // Generate tokens until we reach max_decode_len or the end of the prompt
            std::string response;
            int current_decode_len = 0;
            while (current_decode_len < max_decode_len && !response.empty()) {
                response = llm->response(prompt);
                prompt_len += llm->prompt_len_;
                current_decode_len += llm->gen_seq_len_;
                decode_len += llm->gen_seq_len_;
                prefill_time += llm->prefill_us_;
                decode_time += llm->decode_us_;
                if (current_decode_len >= max_decode_len) {
                    break;
                }
                // Reset the model after each response if needed
                // llm->reset();
            }
            // Reset the model after each response if needed
            llm->reset();
        }
    }
    float prefill_s = prefill_time / 1e6;
    float decode_s = decode_time / 1e6;
    printf("\n#################################\n");
    printf("prompt tokens num  = %d\n", prompt_len);
    printf("decode tokens num  = %d\n", decode_len);
    printf("prefill time = %.2f s\n", prefill_s);
    printf(" decode time = %.2f s\n", decode_s);
    printf("prefill speed = %.2f tok/s\n", prompt_len / prefill_s);
    printf(" decode speed = %.2f tok/s\n", decode_len / decode_s);
    printf("##################################\n");
}

int main(int argc, const char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " model_dir <prompt.txt> <max_decode_len> <num_tests>" << std::endl;
        return 0;
    }
    std::string model_dir = argv[1];
    std::cout << "model path is " << model_dir << std::endl;
    std::unique_ptr<Llm> llm(Llm::createLLM(model_dir));
    llm->load(model_dir);
    int max_decode_len = std::stoi(argv[2]);
    int num_tests = std::stoi(argv[3]);
    std::string prompt_file = argv[4]; // Adjusted argument index
    benchmark(llm.get(), prompt_file, max_decode_len, num_tests);
    return 0;
}