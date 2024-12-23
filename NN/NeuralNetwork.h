#pragma once
#include <vector>
#include <random>
#include <math.h>
#include <thread>

using std::vector;
using std::thread;

enum ActivationFunction {ReLU, Sigmoid, SoftMax};

class NeuralNetwork
{
private:
	size_t layers_count, threads_count;
	vector<size_t> neurons_per_layer;
	vector<vector<float>> layers;
	vector<vector<vector<float>>> weights;
	ActivationFunction act, llact;

public:
	NeuralNetwork();
	int Init(vector<size_t> npl, ActivationFunction act, ActivationFunction llact);
	void PrintLayers();
	void PrintWeights();
	void NeuralMultiplication(vector<float>);
	void Activation(size_t layer, ActivationFunction);
};

