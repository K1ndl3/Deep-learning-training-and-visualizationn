#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <random>
#include <string>
#include <vector>

#define REVERSE_BYTES(x) __builtin_bswap32(x)
#define LEARNING_RATE .01

constexpr int INPUT_SIZE = 784;
constexpr int NUM_EPOCH = 6;

void saveVector(std::ostream& out,const std::vector<double>& vec);
void saveMatrix(std::ostream& out, const std::vector<std::vector<double>>& matrix);


bool validateFileOpen(const std::ifstream& input);

void randomizeVector(std::vector<double>& input);
void randomizedMatrix(std::vector<std::vector<double>>& input);

double sigmoid(double z);

std::vector<double> gradientCalc(std::vector<double>& a,std::vector<std::vector<double>>& weight,std::vector<double> prev_layer, double label, double r);
std::vector<double> gradientBias(std::vector<double>& a, std::vector<double>& bias, double label, double r);
std::vector<double> backpropLayer(
    const std::vector<double>& delta_in,
    const std::vector<std::vector<double>>& W_above,
    std::vector<std::vector<double>>& W_curr,
    std::vector<double>& bias,
    const std::vector<double>& a_curr,
    const std::vector<double>& a_prev,
    double lr);
std::vector<double> forwardPass(std::vector<double> layer,
                                std::vector<std::vector<double>> weight, std::vector<double> bias,bool final = false);

int main() {
    const std::string train_label = "../dataset/train-labels.idx1-ubyte";
    const std::string train_images = "../dataset/train-images.idx3-ubyte";

    std::ifstream label_data(train_label, std::ios::binary);
    std::ifstream image_data(train_images, std::ios::binary);

    if (!validateFileOpen(image_data) || !validateFileOpen(label_data)) {
        std::cerr << "Cannot open image/label files\n";
        return 1;
    }

    // parse label header
    uint32_t magic;
    uint32_t count;
    label_data.read(reinterpret_cast<char*>(&magic), 4);
    label_data.read(reinterpret_cast<char*>(&count), 4);
    magic = REVERSE_BYTES(magic);
    count = REVERSE_BYTES(count);

    std::vector<uint8_t> label_vector(count, 0);
    for (uint32_t i = 0; i < count; i++) {
        uint8_t label;
        label_data.read(reinterpret_cast<char*>(&label), 1);
        label_vector[i] = label;
    }

    // parse image header
    uint32_t num_images;
    uint32_t num_rows;
    uint32_t num_cols;

    image_data.read(reinterpret_cast<char*>(&magic), 4);
    image_data.read(reinterpret_cast<char*>(&num_images), 4);
    image_data.read(reinterpret_cast<char*>(&num_rows), 4);
    image_data.read(reinterpret_cast<char*>(&num_cols), 4);

    num_images = REVERSE_BYTES(num_images);
    num_rows = REVERSE_BYTES(num_rows);
    num_cols = REVERSE_BYTES(num_cols);

    const uint32_t pixels_per_image = num_rows * num_cols;

    std::vector<std::vector<uint8_t>> image_vector(
        num_images, std::vector<uint8_t>(pixels_per_image, 0));

    uint8_t pixel_value;
    for (uint32_t image = 0; image < num_images; image++) {
        for (uint32_t row = 0; row < num_rows; row++) {
            for (uint32_t col = 0; col < num_cols; col++) {
                image_data.read(reinterpret_cast<char*>(&pixel_value), 1);
                image_vector[image][row * num_cols + col] = pixel_value;
            }
        }
    }

    // x[image][pixel] — normalized inputs for the neural network
    std::vector<std::vector<double>> x(
        num_images, std::vector<double>(pixels_per_image, 0.0));

    for (uint32_t im = 0; im < num_images; im++) {
        for (uint32_t i = 0; i < pixels_per_image; i++) {
            x[im][i] = static_cast<double>(image_vector[im][i]) / 255.0;
        }
    }

    // neural network code goes here
    // x[im] is one flattened 28x28 image (784 values in [0, 1])
    // label_vector[im] is the digit 0-9 for that image


    std::vector<double> a1(100, 0.0);
    std::vector<std::vector<double>> w1(100, std::vector<double>(784, 0.0));
    std::vector<double> b1(100,0.0);
    std::vector<double> delta1(100, 0.0);  // matches a1 / w1 rows

    std::vector<double> a2(50, 0.0);
    std::vector<std::vector<double>> w2(50, std::vector<double>(100, 0.0));
    std::vector<double> b2(50,0.0);
    std::vector<double> delta2(50, 0.0);   // matches a2 / w2 rows

    std::vector<double> a3(10, 0.0);
    std::vector<std::vector<double>> w3(10, std::vector<double>(50, 0.0));
    std::vector<double> b3(10,0.0);
    std::vector<double> delta3w(10,0.0);
    std::vector<double> delta3b(10,0.0);


    bool final_layer = true;


    randomizedMatrix(w1);
    randomizeVector(b1);
    
    randomizedMatrix(w2);
    randomizeVector(b2);

    randomizedMatrix(w3);
    randomizeVector(b3);

    for (int epoch = 0; epoch < NUM_EPOCH; epoch++) {
        std::cout << "Epoch: " << epoch << "\n\n\n";
        for (int im = 0; im < 60000; im++) {
        std::cout << "image: " << im <<'\n';

            int y = label_vector[im];
            // layer1
            a1 = forwardPass(x[im], w1, b1);
            // layer2
            a2 = forwardPass(a1, w2, b2);
            // output
            a3 = forwardPass(a2, w3, b3, final_layer);

            //gradient descent and backprop
            delta3w = gradientCalc(a3, w3, a2, double(y), LEARNING_RATE);
            delta3b = gradientBias(a3, b3, y, LEARNING_RATE);

            delta2 = backpropLayer(delta3w, w3, w2, b2, a2, a1, LEARNING_RATE);
            delta1 = backpropLayer(delta2, w2, w1, b1, a1, x[im], LEARNING_RATE);
        }
    }
    std::ofstream outFile("./trained_params.txt");

    saveMatrix(outFile, w1);
    saveVector(outFile, b1);


    saveMatrix(outFile, w2);
    saveVector(outFile, b2);


    saveMatrix(outFile, w3);
    saveVector(outFile, b3);

    return 0;
}

bool validateFileOpen(const std::ifstream& input) {
    return input.is_open();
}

void randomizeVector(std::vector<double>& input) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.1, 0.1);

    for (auto& num : input) {
        num = dis(gen);
    }
}

void randomizedMatrix(std::vector<std::vector<double>>& input) {
    for (auto& row : input) {
        randomizeVector(row);
    }
}

std::vector<double> forwardPass(std::vector<double> layer,
    std::vector<std::vector<double>> weight, std::vector<double> bias, bool final) {

        std::vector<double> h1(weight.size(),0.0);
        for(int row = 0; row < weight.size(); row++) {
            double w_sum = 0.0;
            for(int col = 0; col < weight[0].size(); col++) {
                w_sum += weight[row][col] * layer[col];
            }
            double z = w_sum + bias[row];
            if (!final) {
                h1[row] = sigmoid(z);
            } else {
                h1[row] = z;
            }
        }
        return h1;
    }

double sigmoid(double z) {
    return 1 / (1 + std::exp(-z));
}

std::vector<double> gradientCalc(std::vector<double>& a, std::vector<std::vector<double>>& weight, std::vector<double> prev_layer ,double label, double r) {
    std::vector<double> grad(a.size(), 0.0);
    // this part finishes dl/da3
    for (int i = 0; i < a.size(); i++) {
        double y_val = (i == label) ? 1.0 : 0.0;
        grad[i] = a[i] - y_val;
    }

    // this part will do the outer product
    for (int row = 0; row < weight.size(); row++) {
        for (int col = 0; col < weight[0].size(); col++) {
            weight[row][col] -= r * grad[row] * prev_layer[col];
        }
    }
    return grad;
}

std::vector<double> gradientBias(std::vector<double>& a, std::vector<double>& bias, double label, double r) {
    std::vector<double> grad(a.size(), 0.0);
    // this part finishes dl/da3
    for (int i = 0; i < a.size(); i++) {
        double y_val = (i == label) ? 1.0 : 0.0;
        grad[i] = a[i] - y_val;
    }
    for (int i = 0; i < bias.size(); i++) {
        bias[i] -= r * grad[i];
    }

    return grad;
}

std::vector<double> backpropLayer(
    const std::vector<double>& delta_in,
    const std::vector<std::vector<double>>& W_above,
    std::vector<std::vector<double>>& W_curr,
    std::vector<double>& bias,
    const std::vector<double>& a_curr,
    const std::vector<double>& a_prev,
    double lr) {
        std::vector<double> delta_out(W_curr.size(), 0.0);
        // Step 1: propagate delta backward through W_above
        // delta_out[j] = sum_k delta_in[k] * W_above[k][j]
        for (int j = 0; j < W_curr.size(); j++) {
            double sum = 0.0;
            for (int k = 0; k < delta_in.size(); k++) {
                sum += delta_in[k] * W_above[k][j];
            }
            // Step 2: through sigmoid derivative
            delta_out[j] = sum * a_curr[j] * (1.0 - a_curr[j]);
        }
        // Step 3: update W_curr and bias (same pattern as gradientCalc)
        for (int j = 0; j < W_curr.size(); j++) {
            bias[j] -= lr * delta_out[j];
            for (int i = 0; i < W_curr[0].size(); i++) {
                W_curr[j][i] -= lr * delta_out[j] * a_prev[i];
            }
        }
        return delta_out;
}

void saveVector(std::ostream& out, const std::vector<double>& vec) {
    out << vec.size() << '\n';
    for (int i = 0; i < vec.size(); i++) {
        if (i != 0) {
            out << ' ';
        }
        out << std::setprecision(17) << vec[i];
    }
    out << '\n';
}

void saveMatrix(std::ostream& out, const std::vector<std::vector<double>>& matrix) {
    out << matrix.size() << ' ' << matrix[0].size() << '\n';
    for (const auto& row : matrix) {
        for (size_t i = 0; i < row.size(); i++) {
            if (i > 0) {
                out << ' ';
            }
            out << std::setprecision(17) << row[i];
        }
        out << '\n';
    }
}

