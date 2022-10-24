//
// Created by Mario Jacobo Rios Gamboa on 21/10/22.
//

#pragma once
#include "layer.h"


class MLP {
    vec<Layer> layers;
    void_func<double> activation;
    void_func<double> activation_derivative;
public:
    MLP(const vec<double>& array_init, void_func<double> activation, void_func<double> activation_derivative);
    vec<double> predict(vec<double> input);
    void train(vec<double> input, vec<double> target, const float& rate);
};
