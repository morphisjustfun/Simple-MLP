//
// Created by Mario Jacobo Rios Gamboa on 21/10/22.
//

#pragma once

#include "utils.h"

class Layer {
    vec2<double> weights;
    vec<double> out;

public:
    Layer(const vec2<double> &weights);

    vec<double> get_out() const;
    void set_out(vec<double> out);

    unsigned get_n_neurons() const;
    unsigned get_n_features() const;

    vec2<double> get_weights() const;
};
