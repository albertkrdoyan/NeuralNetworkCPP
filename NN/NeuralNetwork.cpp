#include "NeuralNetwork.h"

float SigmoidFunction(float);
void SoftMaxFunction(vector<float>&, size_t);

int NeuralNetwork::Init(vector<int> npl, ActivationFunction act, ActivationFunction llact) {
	neurons_per_layer = npl;
	layers_count = neurons_per_layer.size();

	layers = vector<vector<float>>(layers_count);

	for (size_t i = 0; i < layers_count; ++i)
		layers[i] = vector<float>(neurons_per_layer[i], 0);

	weights = vector<vector<vector<float>>>(layers_count - 1);

	for (size_t i = 0; i < layers_count - 1; ++i)
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

	this->act = act;
	this->llact = llact;

	return 0;
}

void NeuralNetwork::PrintLayers() {
	for (size_t i = 0; i < layers_count; ++i) {
		printf("Layer [%zu]: ", i);
		for (const auto& neuron : layers[i])
			printf("%f ", neuron);
		printf("\n");
	}
}

void NeuralNetwork::PrintWeights() {
	for (size_t i = 0; i < layers_count - 1; ++i) {
		printf("Weight [%zu]\n[\n", i);
		for (const auto& line : weights[i]) {
			printf("\t[");
			for (const auto& weight : line)
				printf(" %f", weight);
			printf(" ]\n");
		}
		printf("]\n");
	}
}

void NeuralNetwork::NeuralMultiplication(vector<float> fln) {
	if (fln.size() != layers[0].size()) return;

	size_t i = 0, j = 0, k = 0;

	for (i = 0; i < fln.size(); ++i)
		layers[0][i] = fln[i];

	for (i = 1; i < layers_count; ++i) {
		for (size_t j = 0; j < neurons_per_layer[i]; ++j) {
			layers[i][j] = 0.f;

			for (size_t k = 0; k < neurons_per_layer[i - 1]; ++k) {
				layers[i][j] += layers[i - 1][k] * weights[i - 1][j][k];
			}

			layers[i][j] += weights[i - 1][j][neurons_per_layer[i - 1]];
		}

		Activation(i, (i != layers_count - 1 ? act : llact));
	}
}

void NeuralNetwork::Activation(size_t layer, ActivationFunction act) {
	size_t i = 0;
	switch (act)
	{
	case ReLU:
		for (i = 0; i < neurons_per_layer[layer]; ++i)
			if (layers[layer][i] < 0) layers[layer][i] = 0;
		break;
	case Sigmoid:
		for (i = 0; i < neurons_per_layer[layer]; ++i)
			layers[layer][i] = SigmoidFunction(layers[layer][i]);
		break;
	case SoftMax:
		SoftMaxFunction(layers[layer], layers[layer].size());
		break;
	default:
		break;
	}
}

float SigmoidFunction(float x) {
	return 1 / (1 + exp(-x));
}

void SoftMaxFunction(vector<float> &layer, size_t len) {
	// softmax function by ChatGPT
	size_t i;
	float m, sum, constant;

	m = -INFINITY;
	for (i = 0; i < len; ++i) {
		if (m < layer[i]) {
			m = layer[i];
		}
	}

	sum = 0.0f;
	for (i = 0; i < len; ++i) {
		sum += exp(layer[i] - m);
	}

	constant = m + log(sum);
	for (i = 0; i < len; ++i) {
		layer[i] = exp(layer[i] - constant);
	}
}