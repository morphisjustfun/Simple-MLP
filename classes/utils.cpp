//
// Created by Mario Jacobo Rios Gamboa on 23/10/22.
//
#include "random"
#include "utils.h"

double rand01() {
    auto rd = std::random_device{};
    auto gen = std::mt19937{rd()};
    auto dist = std::uniform_real_distribution<>{0, 1};
    return dist(gen);
}

vec<double> random_vec(unsigned int size) {
    vec<double> vec(size);
    for (int i = 0; i < size; ++i) {
        vec[i] = rand01();
    }
    return vec;
}

