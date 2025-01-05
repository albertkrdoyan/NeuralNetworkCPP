#pragma once
#include "addit.h"

enum ActivationFunction {Linear, ReLU, Sigmoid, SoftMax};
enum LossFunction {CrossEntropy, SquaredError};
enum Optimizer { Adam, GradientDescent };

class NeuralNetwork
{
private:
	size_t layers_count, threads_count;
	size_t* neurons_per_layer;
	double** layers;
	double** glayers;
	double*** weights;
	double*** gradients;
	double*** moment1;
	double*** moment2;
	ActivationFunction act, llact;
	LossFunction loss;
	Optimizer opt;
	vector<double> errors;
	double betta1, betta2;
	int t;
public:
	NeuralNetwork();
	~NeuralNetwork();
	int Init(vector<size_t> npl, ActivationFunction act, ActivationFunction llact, LossFunction loss, Optimizer opt); //
	void PrintLayers(size_t); //
	void PrintWeights(); //
	void PrintGradients(const char*, size_t); //
	void PrintInfo(); //
	void NeuralMultiplication(double* input, size_t fln_size);
	void Activation(size_t layer, ActivationFunction);
	void BackProp(vector<float>& y, bool cfl = false);
	void LoadWeights(const char*);
	void SaveWeights(const char*);
	void Optimizing(float, float);
	void Train(vector<vector<float>>&, vector<vector<float>>&, int, size_t, float);
	vector<float> GetLastLayer();
};