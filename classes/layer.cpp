//
// Created by Mario Jacobo Rios Gamboa on 21/10/22.
//

#include <cassert>
#include "layer.h"

Layer::Layer(const vec2<double> &weights) {
    this->weights = weights;
}

vec<double> Layer::get_out() const {
    return this->out;
}

void Layer::set_out(vec<double> out) {
    this->out = out;
}

unsigned Layer::get_n_neurons() const {
    return this->weights.size();
}

unsigned Layer::get_n_features() const {
    assert (!this->weights.empty());
    return this->weights[0].size();
}

vec2<double> Layer::get_weights() const {
    return this->weights;
}
