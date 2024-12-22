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
	size_t layers_count;
	vector<int> neurons_per_layer;
	vector<vector<float>> layers;
	vector<vector<vector<float>>> weights;
	ActivationFunction act, llact;

	int threads_count;

public:
	int Init(vector<int> npl, ActivationFunction act, ActivationFunction llact, int tc = 5);
	void PrintLayers();
	void PrintWeights();
	void NeuralMultiplication(vector<float>);
	void Activation(size_t layer, ActivationFunction);
	void NMPThread(int, int, int);
};

