#include "NeuralNetwork.h"

float SigmoidFunction(float);
void SoftMaxFunction(vector<float>&, size_t);
void NMPThread(size_t, size_t, size_t, vector<vector<float>>&, vector<size_t>&, vector<vector<vector<float>>>&);

NeuralNetwork::NeuralNetwork() {
	this->threads_count = 1;
	this->llact = act = ReLU;
	this->layers_count = 0;
}

int NeuralNetwork::Init(vector<size_t> npl, ActivationFunction act, ActivationFunction llact, LossFunction loss) {
	neurons_per_layer = npl;
	layers_count = neurons_per_layer.size();

	layers = vector<vector<float>>(layers_count);
	glayers = vector<vector<float>>(layers_count);

	for (size_t i = 0; i < layers_count; ++i) {
		layers[i] = vector<float>(neurons_per_layer[i], 0);
		glayers[i] = vector<float>(neurons_per_layer[i], 0);
	}

	weights = vector<vector<vector<float>>>(layers_count - 1);
	gradients = vector<vector<vector<float>>>(layers_count - 1);

	for (size_t i = 0; i < layers_count - 1; ++i) {
		weights[i] = vector<vector<float>>(neurons_per_layer[i + 1], vector<float>(neurons_per_layer[i] + 1, 0.f));
		gradients[i] = vector<vector<float>>(neurons_per_layer[i + 1], vector<float>(neurons_per_layer[i] + 1, 0.f));
	}

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
	this->loss = loss;

	return 0;
}

void NeuralNetwork::PrintLayers(size_t layer) {
	for (size_t i = layer; i < (layer == 0 ? layers_count : layer + 1); ++i) {
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

void NeuralNetwork::PrintGradients(const char* printwhat, size_t layer)
{
	if (printwhat == "ALL" || printwhat == "GLAYER") {
		for (size_t i = layer; i < (layer == 0 ? layers_count : layer + 1); ++i) {
			printf("GLayer [%zu]: ", i);
			for (const auto& neuron : glayers[i])
				printf("%f ", neuron);
			printf("\n");
		}
	}
	if (printwhat == "ALL" || printwhat == "GRAD") {
		for (size_t i = 0; i < layers_count - 1; ++i) {
			printf("Gradients [%zu]\n[\n", i);
			for (const auto& line : gradients[i]) {
				printf("\t[");
				for (const auto& weight : line)
					printf(" %f", weight);
				printf(" ]\n");
			}
			printf("]\n");
		}
	}
}

void NeuralNetwork::NeuralMultiplicationT(vector<float> fln) {
	if (fln.size() != layers[0].size()) return;

	size_t i = 0, j = 0, k = 0;

	for (i = 0; i < fln.size(); ++i)
		layers[0][i] = fln[i];	

	for (i = 1; i < layers_count; ++i) {
		size_t c_t_c = (neurons_per_layer[i] < 64 ? 1 : threads_count);
		vector<thread> threads(c_t_c);

		for (size_t t_i = 0; t_i < c_t_c; ++t_i) {
			threads[t_i] = thread(
				NMPThread, 
				i, 
				t_i * (neurons_per_layer[i] / c_t_c),
				(t_i != c_t_c - 1 ? (t_i + 1) * (neurons_per_layer[i] / c_t_c) : neurons_per_layer[i]),
				std::ref(layers),
				std::ref(neurons_per_layer),
				std::ref(weights)
			);
		}

		for (auto& t : threads)
			t.join();
		
		Activation(i, (i != layers_count - 1 ? act : llact));
	}
}

void NeuralNetwork::NeuralMultiplication(vector<float> fln) {
	if (fln.size() != layers[0].size()) return;

	size_t i = 0, j = 0, k = 0;

	for (i = 0; i < fln.size(); ++i)
		layers[0][i] = fln[i];

	for (i = 1; i < layers_count; ++i) {
		//#pragma omp parallel for
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

void NeuralNetwork::BackProp(vector<float> y, bool calculate_first_layer)
{
	if (y.size() != layers.back().size()) return;
	
	if (loss == LossFunction::CrossEntropy && llact == ActivationFunction::SoftMax) {
		for (size_t i = 0; i < neurons_per_layer.back(); ++i) {
			// dE / dz
			glayers.back()[i] = layers.back()[i] - y[i];
		}
	}
	else {
		for (size_t i = 0; i < neurons_per_layer.back(); ++i) {
			// dE / dz
			glayers.back()[i] = 1.0f;
			if (loss == LossFunction::SquaredError)
				glayers.back()[i] *= 2 * (layers.back()[i] - y[i]);

			if (llact == ActivationFunction::ReLU)
				glayers.back()[i] *= (layers.back()[i] > 0 ? layers.back()[i] : 0);
			else if (llact == ActivationFunction::Sigmoid)
				glayers.back()[i] *= layers.back()[i] * (1 - layers.back()[i]);
		}
	}

	for (int n = layers_count - 2; n >= 0; --n) {
		size_t j = 0, i = 0;
		vector<float> deda(weights[n][j].size() - 1, 0);
		
		for (j = 0; j < weights[n].size(); ++j) {
			for (i = 0; i < weights[n][j].size() - 1; ++i) {
				gradients[n][j][i] += glayers[n + 1][j] * layers[n][i];
				deda[i] += weights[n][j][i];
			}
			gradients[n][j][i] = glayers[n + 1][j];
		}

		for (i = 0; n != 0 && i < deda.size(); ++i)
			glayers[n][i] = deda[i];
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

void NMPThread(size_t i, size_t st, size_t end, vector<vector<float>>& layers, vector<size_t>& neurons_per_layer, vector<vector<vector<float>>>& weights) {
	for (size_t j = st; j < end; ++j) {
		layers[i][j] = 0.f;

		for (size_t k = 0; k < neurons_per_layer[i - 1]; ++k) {
			layers[i][j] += layers[i - 1][k] * weights[i - 1][j][k];
		}

		layers[i][j] += weights[i - 1][j][neurons_per_layer[i - 1]];
	}
}