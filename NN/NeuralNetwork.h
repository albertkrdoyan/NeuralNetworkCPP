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
	int Init(vector<int> npl);
	void PrintLayers();
	void PrintWeights();
};

