#pragma once
#include <vector>
#include <random>
#include <math.h>
#include <thread>
#include <omp.h>
#include <fstream>
#include <chrono>
#include <time.h>

using std::vector;
using std::thread;
using std::ifstream;
using std::ofstream;

void plot(vector<float>);
template <class T>
void Shuffle(vector<T>&, vector<T>&);

void LoadX(vector<vector<float>>& X, const char* sourcePath);
void LoadY(vector<vector<float>>& Y, const char* sourcePath);

enum ActivationFunction {Linear, ReLU, Sigmoid, SoftMax};
enum LossFunction {CrossEntropy, SquaredError};
enum Optimizer { Adam, GradientDescent };

class NeuralNetwork
{
private:
	size_t layers_count, threads_count;
	vector<size_t> neurons_per_layer;
	vector<vector<float>> layers;
	vector<vector<float>> glayers;	
	vector<vector<vector<float>>> weights;
	vector<vector<vector<float>>> gradients;
	vector<vector<vector<float>>> moment1;
	vector<vector<vector<float>>> moment2;
	ActivationFunction act, llact;
	LossFunction loss;
	Optimizer opt;
	vector<float> errors;
	float betta1, betta2;
	int t;
public:
	NeuralNetwork();
	int Init(vector<size_t> npl, ActivationFunction act, ActivationFunction llact, LossFunction loss, Optimizer opt);
	void PrintLayers(size_t);
	void PrintWeights();
	void PrintGradients(const char*, size_t);
	void PrintInfo();
	void NeuralMultiplication(vector<float>&);
	void Activation(size_t layer, ActivationFunction);
	void BackProp(vector<float>& y, bool cfl = false);
	void LoadWeights(const char*);
	void SaveWeights(const char*);
	void ResetGradients();
	void Optimizing(float, float);
	void Train(vector<vector<float>>&, vector<vector<float>>&, int, size_t, float);
	vector<float> GetLastLayer();
};