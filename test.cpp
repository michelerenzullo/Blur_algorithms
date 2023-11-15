#include "Utils.hpp"
#include <random>
#include <algorithm>
#include "FastBoxBlur/fast_box_blur.h"
std::mutex mtx;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);



void test_threading(int iterations) {
    std::cout << std::thread::hardware_concurrency() << " Threads and " << iterations << " iterations\n";
    hybrid_loop(iterations, [&](auto n, auto thread_id) {
        mtx.lock();
        std::cout << "Thread: " << thread_id << ", res: " << n << "\n";
        mtx.unlock();
    });
    printf("\n");
}

void test_flip(int width, int height) {
    auto input_data = std::make_unique<float[]>(width * height);
    auto output_data = std::make_unique<float[]>(width * height);
    std::generate(input_data.get(), input_data.get() + width * height, [&]() { return dis(gen); });
    uint32_t input_crc = crc32c((uint8_t*)(input_data.get()), width * height * sizeof(float));

    std::chrono::time_point<std::chrono::steady_clock> start_time = std::chrono::steady_clock::now();
    flip_block<float, 1>(input_data.get(), output_data.get(), height, width);
    std::cout << "transpose ms: " << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count() << "\n";

    auto input_data_copy = std::make_unique<float[]>(width * height);
    std::copy_n(input_data.get(), width * height, input_data_copy.get());
    std::fill_n(input_data.get(), width * height, 0.0f);
    flip_block<float, 1>(output_data.get(), input_data.get(), width, height);
    uint32_t output_crc = crc32c((uint8_t*)(input_data.get()), width * height * sizeof(float));

    printf(input_crc == output_crc ? "Transpose test passed, crc32: %X %X\n" : "Transpose test failed: %d %d\n", input_crc, output_crc);
}

void test_interleave_deinterleave(int width, int height) {

    auto input_image = std::make_unique<float[]>(width * height * 3);
    std::generate(input_image.get(), input_image.get() + width * height * 3, [&]() { return dis(gen); });
    uint32_t input_crc = crc32c((uint8_t*)(input_image.get()), width * height * 3 * sizeof(float));

    std::vector<std::vector<float>> temp(3, std::vector<float>(width * height));
    float* BGR[3] = { temp[0].data(), temp[1].data(), temp[2].data() };

    auto start = std::chrono::steady_clock::now();
    deinterleave_BGR((const float*)input_image.get(), BGR, width * height);
    auto deinterleave_duration = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Deinterleave duration: " << deinterleave_duration << " ms\n";

    auto output_image = std::make_unique<float[]>(width * height * 3);
    std::fill_n(output_image.get(), width * height * 3, 0.0f);

    start = std::chrono::steady_clock::now();
    interleave_BGR((const float**)BGR, (float*)output_image.get(), width * height);
    auto interleave_duration = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    std::cout << "Interleave duration: " << interleave_duration << " ms\n";
    uint32_t output_crc = crc32c((uint8_t*)(output_image.get()), width * height * 3 * sizeof(float));

    printf(input_crc == output_crc ? "Deinterleave/Interleave test passed, crc32: %X %X\n" : "Deinterleave/Interleave test failed: %d %d\n", input_crc, output_crc);
}


int main(int argc, char *argv[]){

    int iterations = argv[1] ? atoi(argv[1]) : 256;
    int width = argv[1] ? atoi(argv[1]) : 1;
    int height = argv[2] ? atoi(argv[2]) : 1;

    test_threading(iterations);
    test_flip(width, height);
    test_interleave_deinterleave(width, height);
}
