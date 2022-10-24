#include <iostream>
#include <valarray>
#include <random>
#include "classes/mlp.h"

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double derivative_sigmoid(double x) {
    return sigmoid(x) * (1.0 - sigmoid(x));
}


double ReLU(double x) {
    return std::max(0.0, x);
}

double derivative_ReLU(double x) {
    return x > 0 ? 1.0 : 0.0;
}


int main() {
    auto iris = vec2<double>(150);
    for (int i = 0; i < 150; ++i) {
        iris[i] = vec<double>(5);
    }
    iris[0] = {5.1, 3.5, 1.4, 0.2, 0};
    iris[1] = {4.9, 3.0, 1.4, 0.2, 0};
    iris[2] = {4.7, 3.2, 1.3, 0.2, 0};
    iris[3] = {4.6, 3.1, 1.5, 0.2, 0};
    iris[4] = {5.0, 3.6, 1.4, 0.2, 0};
    iris[5] = {5.4, 3.9, 1.7, 0.4, 0};
    iris[6] = {4.6, 3.4, 1.4, 0.3, 0};
    iris[7] = {5.0, 3.4, 1.5, 0.2, 0};
    iris[8] = {4.4, 2.9, 1.4, 0.2, 0};
    iris[9] = {4.9, 3.1, 1.5, 0.1, 0};
    iris[10] = {5.4, 3.7, 1.5, 0.2, 0};
    iris[11] = {4.8, 3.4, 1.6, 0.2, 0};
    iris[12] = {4.8, 3.0, 1.4, 0.1, 0};
    iris[13] = {4.3, 3.0, 1.1, 0.1, 0};
    iris[14] = {5.8, 4.0, 1.2, 0.2, 0};
    iris[15] = {5.7, 4.4, 1.5, 0.4, 0};
    iris[16] = {5.4, 3.9, 1.3, 0.4, 0};
    iris[17] = {5.1, 3.5, 1.4, 0.3, 0};
    iris[18] = {5.7, 3.8, 1.7, 0.3, 0};
    iris[19] = {5.1, 3.8, 1.5, 0.3, 0};
    iris[20] = {5.4, 3.4, 1.7, 0.2, 0};
    iris[21] = {5.1, 3.7, 1.5, 0.4, 0};
    iris[22] = {4.6, 3.6, 1.0, 0.2, 0};
    iris[23] = {5.1, 3.3, 1.7, 0.5, 0};
    iris[24] = {4.8, 3.4, 1.9, 0.2, 0};
    iris[25] = {5.0, 3.0, 1.6, 0.2, 0};
    iris[26] = {5.0, 3.4, 1.6, 0.4, 0};
    iris[27] = {5.2, 3.5, 1.5, 0.2, 0};
    iris[28] = {5.2, 3.4, 1.4, 0.2, 0};
    iris[29] = {4.7, 3.2, 1.6, 0.2, 0};
    iris[30] = {4.8, 3.1, 1.6, 0.2, 0};
    iris[31] = {5.4, 3.4, 1.5, 0.4, 0};
    iris[32] = {5.2, 4.1, 1.5, 0.1, 0};
    iris[33] = {5.5, 4.2, 1.4, 0.2, 0};
    iris[34] = {4.9, 3.1, 1.5, 0.1, 0};
    iris[35] = {5.0, 3.2, 1.2, 0.2, 0};
    iris[36] = {5.5, 3.5, 1.3, 0.2, 0};
    iris[37] = {4.9, 3.1, 1.5, 0.1, 0};
    iris[38] = {4.4, 3.0, 1.3, 0.2, 0};
    iris[39] = {5.1, 3.4, 1.5, 0.2, 0};
    iris[40] = {5.0, 3.5, 1.3, 0.3, 0};
    iris[41] = {4.5, 2.3, 1.3, 0.3, 0};
    iris[42] = {4.4, 3.2, 1.3, 0.2, 0};
    iris[43] = {5.0, 3.5, 1.6, 0.6, 0};
    iris[44] = {5.1, 3.8, 1.9, 0.4, 0};
    iris[45] = {4.8, 3.0, 1.4, 0.3, 0};
    iris[46] = {5.1, 3.8, 1.6, 0.2, 0};
    iris[47] = {4.6, 3.2, 1.4, 0.2, 0};
    iris[48] = {5.3, 3.7, 1.5, 0.2, 0};
    iris[49] = {5.0, 3.3, 1.4, 0.2, 0};
    iris[50] = {7.0, 3.2, 4.7, 1.4, 1};
    iris[51] = {6.4, 3.2, 4.5, 1.5, 1};
    iris[52] = {6.9, 3.1, 4.9, 1.5, 1};
    iris[53] = {5.5, 2.3, 4.0, 1.3, 1};
    iris[54] = {6.5, 2.8, 4.6, 1.5, 1};
    iris[55] = {5.7, 2.8, 4.5, 1.3, 1};
    iris[56] = {6.3, 3.3, 4.7, 1.6, 1};
    iris[57] = {4.9, 2.4, 3.3, 1.0, 1};
    iris[58] = {6.6, 2.9, 4.6, 1.3, 1};
    iris[59] = {5.2, 2.7, 3.9, 1.4, 1};
    iris[60] = {5.0, 2.0, 3.5, 1.0, 1};
    iris[61] = {5.9, 3.0, 4.2, 1.5, 1};
    iris[62] = {6.0, 2.2, 4.0, 1.0, 1};
    iris[63] = {6.1, 2.9, 4.7, 1.4, 1};
    iris[64] = {5.6, 2.9, 3.6, 1.3, 1};
    iris[65] = {6.7, 3.1, 4.4, 1.4, 1};
    iris[66] = {5.6, 3.0, 4.5, 1.5, 1};
    iris[67] = {5.8, 2.7, 4.1, 1.0, 1};
    iris[68] = {6.2, 2.2, 4.5, 1.5, 1};
    iris[69] = {5.6, 2.5, 3.9, 1.1, 1};
    iris[70] = {5.9, 3.2, 4.8, 1.8, 1};
    iris[71] = {6.1, 2.8, 4.0, 1.3, 1};
    iris[72] = {6.3, 2.5, 4.9, 1.5, 1};
    iris[73] = {6.1, 2.8, 4.7, 1.2, 1};
    iris[74] = {6.4, 2.9, 4.3, 1.3, 1};
    iris[75] = {6.6, 3.0, 4.4, 1.4, 1};
    iris[76] = {6.8, 2.8, 4.8, 1.4, 1};
    iris[77] = {6.7, 3.0, 5.0, 1.7, 1};
    iris[78] = {6.0, 2.9, 4.5, 1.5, 1};
    iris[79] = {5.7, 2.6, 3.5, 1.0, 1};
    iris[80] = {5.5, 2.4, 3.8, 1.1, 1};
    iris[81] = {5.5, 2.4, 3.7, 1.0, 1};
    iris[82] = {5.8, 2.7, 3.9, 1.2, 1};
    iris[83] = {6.0, 2.7, 5.1, 1.6, 1};
    iris[84] = {5.4, 3.0, 4.5, 1.5, 1};
    iris[85] = {6.0, 3.4, 4.5, 1.6, 1};
    iris[86] = {6.7, 3.1, 4.7, 1.5, 1};
    iris[87] = {6.3, 2.3, 4.4, 1.3, 1};
    iris[88] = {5.6, 3.0, 4.1, 1.3, 1};
    iris[89] = {5.5, 2.5, 4.0, 1.3, 1};
    iris[90] = {5.5, 2.6, 4.4, 1.2, 1};
    iris[91] = {6.1, 3.0, 4.6, 1.4, 1};
    iris[92] = {5.8, 2.6, 4.0, 1.2, 1};
    iris[93] = {5.0, 2.3, 3.3, 1.0, 1};
    iris[94] = {5.6, 2.7, 4.2, 1.3, 1};
    iris[95] = {5.7, 3.0, 4.2, 1.2, 1};
    iris[96] = {5.7, 2.9, 4.2, 1.3, 1};
    iris[97] = {6.2, 2.9, 4.3, 1.3, 1};
    iris[98] = {5.1, 2.5, 3.0, 1.1, 1};
    iris[99] = {5.7, 2.8, 4.1, 1.3, 1};
    iris[100] = {6.3, 3.3, 6.0, 2.5, 2};
    iris[101] = {5.8, 2.7, 5.1, 1.9, 2};
    iris[102] = {7.1, 3.0, 5.9, 2.1, 2};
    iris[103] = {6.3, 2.9, 5.6, 1.8, 2};
    iris[104] = {6.5, 3.0, 5.8, 2.2, 2};
    iris[105] = {7.6, 3.0, 6.6, 2.1, 2};
    iris[106] = {4.9, 2.5, 4.5, 1.7, 2};
    iris[107] = {7.3, 2.9, 6.3, 1.8, 2};
    iris[108] = {6.7, 2.5, 5.8, 1.8, 2};
    iris[109] = {7.2, 3.6, 6.1, 2.5, 2};
    iris[110] = {6.5, 3.2, 5.1, 2.0, 2};
    iris[111] = {6.4, 2.7, 5.3, 1.9, 2};
    iris[112] = {6.8, 3.0, 5.5, 2.1, 2};
    iris[113] = {5.7, 2.5, 5.0, 2.0, 2};
    iris[114] = {5.8, 2.8, 5.1, 2.4, 2};
    iris[115] = {6.4, 3.2, 5.3, 2.3, 2};
    iris[116] = {6.5, 3.0, 5.5, 1.8, 2};
    iris[117] = {7.7, 3.8, 6.7, 2.2, 2};
    iris[118] = {7.7, 2.6, 6.9, 2.3, 2};
    iris[119] = {6.0, 2.2, 5.0, 1.5, 2};
    iris[120] = {6.9, 3.2, 5.7, 2.3, 2};
    iris[121] = {5.6, 2.8, 4.9, 2.0, 2};
    iris[122] = {7.7, 2.8, 6.7, 2.0, 2};
    iris[123] = {6.3, 2.7, 4.9, 1.8, 2};
    iris[124] = {6.7, 3.3, 5.7, 2.1, 2};
    iris[125] = {7.2, 3.2, 6.0, 1.8, 2};
    iris[126] = {6.2, 2.8, 4.8, 1.8, 2};
    iris[127] = {6.1, 3.0, 4.9, 1.8, 2};
    iris[128] = {6.4, 2.8, 5.6, 2.1, 2};
    iris[129] = {7.2, 3.0, 5.8, 1.6, 2};
    iris[130] = {7.4, 2.8, 6.1, 1.9, 2};
    iris[131] = {7.9, 3.8, 6.4, 2.0, 2};
    iris[132] = {6.4, 2.8, 5.6, 2.2, 2};
    iris[133] = {6.3, 2.8, 5.1, 1.5, 2};
    iris[134] = {6.1, 2.6, 5.6, 1.4, 2};
    iris[135] = {7.7, 3.0, 6.1, 2.3, 2};
    iris[136] = {6.3, 3.4, 5.6, 2.4, 2};
    iris[137] = {6.4, 3.1, 5.5, 1.8, 2};
    iris[138] = {6.0, 3.0, 4.8, 1.8, 2};
    iris[139] = {6.9, 3.1, 5.4, 2.1, 2};
    iris[140] = {6.7, 3.1, 5.6, 2.4, 2};
    iris[141] = {6.9, 3.1, 5.1, 2.3, 2};
    iris[142] = {5.8, 2.7, 5.1, 1.9, 2};
    iris[143] = {6.8, 3.2, 5.9, 2.3, 2};
    iris[144] = {6.7, 3.3, 5.7, 2.5, 2};
    iris[145] = {6.7, 3.0, 5.2, 2.3, 2};
    iris[146] = {6.3, 2.5, 5.0, 1.9, 2};
    iris[147] = {6.5, 3.0, 5.2, 2.0, 2};
    iris[148] = {6.2, 3.4, 5.4, 2.3, 2};
    iris[149] = {5.9, 3.0, 5.1, 1.8, 2};

    // shuffle the data
    std::shuffle(iris.begin(), iris.end(), std::mt19937{std::random_device{}()});
    // split the data into training and testing
    std::vector<std::vector<double>> training_data(100, std::vector<double>(4));
    std::vector<std::vector<double>> testing_data(50, std::vector<double>(4));
    std::vector<int> training_labels(100);
    std::vector<int> testing_labels(50);

    for (int i = 0; i < 100; i++) {
        training_data[i] = {iris[i][0], iris[i][1], iris[i][2], iris[i][3]};
        training_labels[i] = iris[i][4];
    }

    for (int i = 0; i < 50; i++) {
        testing_data[i] = {iris[i + 100][0], iris[i + 100][1], iris[i + 100][2], iris[i + 100][3]};
        testing_labels[i] = iris[i + 100][4];
    }

    // FIRST ARGUMENT = vector containing the number of neurons in each layer
    // SECOND ARGUMENT = activation function
    // THIRD ARGUMENT = derivative of the activation function
    auto mlp_relu = MLP({4, 5, 6, 3, 9, 2, 1}, ReLU, derivative_ReLU);
    for (int i = 0; i < 100; i++) {
        // FIRST ARGUMENT = vector containing the input data
        // SECOND ARGUMENT = vector containing the expected output
        // THIRD ARGUMENT = learning rate
        mlp_relu.train(training_data[i], {static_cast<double>(training_labels[i])}, 0.001);
    }
    int correct = 0;
    for (int i = 0; i < 50; i++) {
        auto predicted = mlp_relu.predict(testing_data[i]);
        int rounded = std::round(predicted[0]);
        if (rounded == testing_labels[i]) {
            correct++;
        }
    }
    std::cout << "Accuracy using a 4-5-6-3-9-2-1 MLP with ReLU activation function: " << correct / 50.0 << std::endl;

    auto mlp_sigmoid = MLP({4, 5, 6, 3, 9, 2, 1}, sigmoid, derivative_sigmoid);
    for (int i = 0; i < 100; i++) {
        // FIRST ARGUMENT = vector containing the input data
        // SECOND ARGUMENT = vector containing the expected output
        // THIRD ARGUMENT = learning rate
        mlp_sigmoid.train(training_data[i], {static_cast<double>(training_labels[i])}, 0.001);
    }
    correct = 0;
    for (int i = 0; i < 50; i++) {
        auto predicted = mlp_sigmoid.predict(testing_data[i]);
        int rounded = std::round(predicted[0]);
        if (rounded == testing_labels[i]) {
            correct++;
        }
    }
    std::cout << "Accuracy using a 4-5-6-3-9-2-1 MLP with sigmoid activation function: " << correct / 50.0 << std::endl;

    return 0;
}

