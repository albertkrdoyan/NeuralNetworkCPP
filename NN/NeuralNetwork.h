#pragma once
#include <vector>
#include <random>
using std::vector;

class NeuralNetwork
{
private:
	size_t layers_count;
	vector<int> neurons_per_layer;
	vector<vector<float>> layers;
	vector<vector<vector<float>>> weights;

public:
	int Init(vector<int> npl) {
		neurons_per_layer = npl;
		layers_count = neurons_per_layer.size();

		layers = vector<vector<float>>(layers_count);

		for (int i = 0; i < layers_count; ++i)
			layers[i] = vector<float>(neurons_per_layer[i], 0);

		weights = vector<vector<vector<float>>>(layers_count - 1);

		for (int i = 0; i < layers_count - 1; ++i)
			weights[i] = vector<vector<float>>(neurons_per_layer[i + 1], vector<float>(neurons_per_layer[i] + 1, 0.f));

		std::random_device rd; // Seed for the random number engine
		std::mt19937 gen(rd()); // Mersenne Twister random number engine
		std::uniform_real_distribution<> dist(0.0, 1.0); // Range [0, 1)

		for (auto& weight : weights) {
			for (auto& line : weight) {
				for (auto& w : line)
					w = static_cast<float>(dist(gen));
			}
		}

		return 0;
	}

	void PrintLayers() {
		for (int i = 0; i < layers_count; ++i) {
			printf("Layer [%d]: ", i);
			for (const auto& neuron : layers[i])
				printf("%f ", neuron);
			printf("\n");
		}
	}

	void PrintWeights() {
		for (int i = 0; i < layers_count - 1; ++i) {
			printf("Weight [%d]\n[\n", i);
			for (const auto& line : weights[i]) {
				printf("\t[");
				for (const auto& weight : line)
					printf(" %f", weight);
				printf(" ]\n");
			}
			printf("]\n");
		}
	}
};

