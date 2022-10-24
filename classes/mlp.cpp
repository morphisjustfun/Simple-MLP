//
// Created by Mario Jacobo Rios Gamboa on 21/10/22.
//

#include "mlp.h"

#include <utility>
#include <cassert>

MLP::MLP(const vec<double> &array_init, void_func<double> activation, void_func<double> activation_derivative) {
    this->activation = activation;
    this->activation_derivative = activation_derivative;
    for (int i = 1; i < array_init.size(); ++i) {
        auto weights_random = vec2<double>(array_init[i]);
        for (int j = 0; j < array_init[i]; ++j) {
            weights_random[j] = random_vec(array_init[i - 1]);
        }
        this->layers.push_back(Layer(weights_random));
    }
}

vec<double> MLP::predict(vec<double> input) {
    auto current_input = std::move(input);
    for (auto &layer : this->layers) {
        auto current_output = vec<double>(layer.get_n_neurons());
        for (int i = 0; i < layer.get_n_neurons(); ++i) {
            auto current_weight = layer.get_weights()[i];
            auto current_sum = 0.0;
            for (int j = 0; j < layer.get_n_features(); ++j) {
                current_sum -= current_weight[j] * current_input[j];
            }
            current_sum -= current_weight[layer.get_n_features()];
            current_output[i] = this->activation(current_sum);
        }
        layer.set_out(current_output);
        current_input = current_output;
    }
    return current_input;
}

void MLP::train(vec<double> input, vec<double> target, const float& rate) {
    vec<double> output = this->predict(input); // forward pass

    assert (output.size() == target.size());

    // first last layer
    vec<double> delta_last = vec<double>(output.size());

    for (unsigned i = 0; i < output.size(); ++i) {
        delta_last[i] = (target[i] - output[i]) * this->activation_derivative(output[i]);
        for (int j = 0; j < this->layers.back().get_n_features(); ++j) {
            if (j == this->layers.back().get_n_features() - 1) {
                this->layers.back().get_weights()[i][j] -= rate * delta_last[i];
            } else {
                this->layers.back().get_weights()[i][j] -= rate * delta_last[i] * input[j];
            }
        }
    }
    // then hidden layers
    for (int i = this->layers.size() - 2; i >= 0; --i) {
        auto current_layer = this->layers[i];
        auto next_layer = this->layers[i + 1];
        auto delta = vec<double>(current_layer.get_n_neurons());

        for (int j = 0; j < current_layer.get_n_neurons(); ++j) {
            auto current_sum = 0.0;
            for (int k = 0; k < next_layer.get_n_neurons(); ++k) {
                current_sum -= next_layer.get_weights()[k][j] * delta_last[k];
            }
            delta[j] = current_sum * this->activation_derivative(current_layer.get_out()[j]);
            for (int k = 0; k < current_layer.get_n_features(); ++k) {
                if (k == current_layer.get_n_features() - 1) {
                    current_layer.get_weights()[j][k] -= rate * delta[j];
                } else {
                    current_layer.get_weights()[j][k] -= rate * delta[j] * input[k];
                }
            }
        }
    }
}
