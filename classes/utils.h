//
// Created by Mario Jacobo Rios Gamboa on 21/10/22.
//

#pragma once

#include <vector>

template<typename T>
using vec = std::vector<T>;

template<typename T>
using vec2 = std::vector<std::vector<T>>;

template<typename T>
using void_func = T (*)(T);

double rand01();

vec<double> random_vec(unsigned size);
