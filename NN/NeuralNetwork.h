#pragma once
#include <vector>
#include <random>
#include <math.h>
using std::vector;

enum ActivationFunction {ReLU, Sigmoid, SoftMax};

class NeuralNetwork
{
private:
	size_t layers_count;
	vector<int> neurons_per_layer;
	vector<vector<float>> layers;
	vector<vector<vector<float>>> weights;
	ActivationFunction act, llact;

public:
	int Init(vector<int> npl, ActivationFunction act, ActivationFunction llact);
	void PrintLayers();
	void PrintWeights();
	void NeuralMultiplication(vector<float> fln);
	void Activation(size_t layer, ActivationFunction act);
};

