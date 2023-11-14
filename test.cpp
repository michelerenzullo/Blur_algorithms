#include "Utils.hpp"
#include <random>
#include <algorithm>
#include "FastBoxBlur/fast_box_blur.h"
std::mutex mtx;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

auto test = [](const int i){
    printf("%d Threads and %d iterations\n", std::thread::hardware_concurrency(), i);
    hybrid_loop(i, [&](auto n, auto tid) {
        mtx.lock();
        printf("Thread: %d, res: %d\n", tid, n);
        mtx.unlock();
    });
    printf("\n");
};
int main(int argc, char *argv[]){
    int i = argv[1] ? atoi(argv[1]) : 10;
    int w_ = argv[1] ? atoi(argv[1]) : 1;
    int h_ = argv[2] ? atoi(argv[2]) : 1;
    // test(i);

    auto ip = std::make_unique<float[]>(w_ * h_);
    auto op = std::make_unique<float[]>(w_ * h_);
    std::generate(ip.get(), ip.get() + w_ * h_, [&]() { return dis(gen); });

    flip_block<float, 1>(ip.get(), op.get(), h_, w_);
    /*for(int w=0; w< w_; ++w){
        for(int h=0; h<h_; ++h)
            printf("%f ",ip.get()[w * h_ + h]);
        printf("\n");
    }
    printf("\n");
    for(int h=0; h< h_; ++h){
        for(int w=0; w<w_; ++w)
            printf("%f ",op.get()[h * w_ + w]);
        printf("\n");
    }*/

    auto ip_copy = std::make_unique<float[]>(w_ * h_);
    std::copy_n(ip.get(), w_ * h_, ip_copy.get());
    std::fill_n(ip.get(), w_ * h_, 0.0f);
    flip_block<float, 1>(op.get(), ip.get(), w_, h_);
    if (std::equal(ip.get(), ip.get() + w_ * h_, ip_copy.get())) {
        std::cout << "ip_copy data is equal to ip data\n";
    } else {
        std::cout << "ip_copy data is not equal to ip data\n";
    }

    /*printf("\n");
    for(int w = 0; w < w_; ++w){
        for(int h = 0; h < h_; ++h)
            printf("%f ", ip_copy.get()[h * w_ + w]);
        printf("\n");
    }*/
}
