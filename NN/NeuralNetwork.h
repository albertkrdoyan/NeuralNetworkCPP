#pragma once
#include <vector>
#include <random>
#include <math.h>
#include <thread>
#include <omp.h>
#include <fstream>
#include <chrono>

using std::vector;
using std::thread;
using std::ifstream;
using std::ofstream;

enum ActivationFunction {ReLU, Sigmoid, SoftMax};
enum LossFunction {CrossEntropy, SquaredError};
enum Optimizer {Adam, GradientDescent};

class NeuralNetwork
{
private:
	size_t layers_count, threads_count;
	vector<size_t> neurons_per_layer;
	vector<vector<float>> layers;
	vector<vector<float>> glayers;
	vector<vector<vector<float>>> weights;
	vector<vector<vector<float>>> gradients;
	ActivationFunction act, llact;
	LossFunction loss;
	Optimizer opt;
public:
	NeuralNetwork();
	int Init(vector<size_t> npl, ActivationFunction act, ActivationFunction llact, LossFunction loss, Optimizer opt);
	void PrintLayers(size_t);
	void PrintWeights();
	void PrintGradients(const char*, size_t);
	void PrintInfo();
	void NeuralMultiplication(vector<float>&);
	void NeuralMultiplicationT(vector<float>&);
	void Activation(size_t layer, ActivationFunction);
	void BackProp(vector<float>& y, bool cfl = false);
	void LoadWeights(const char*);
	void SaveWeights(const char*);
	void ResetGradients();
	void Optimizing(float);
	void Train(vector<vector<float>>&, vector<vector<float>>&, int, size_t, float);
};

