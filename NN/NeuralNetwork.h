#pragma once
#include <vector>
#include <random>
#include <math.h>
#include <thread>
#include <omp.h>
#include <fstream>
#include <chrono>
#include <time.h>
#include <iomanip>

using std::vector;
using std::thread;
using std::ifstream;
using std::ofstream;

enum ActivationFunction { Linear, ReLU, Sigmoid, SoftMax };
enum LossFunction { CrossEntropy, SquaredError };
enum Optimizer { Adam, GradientDescent };

class DataSet {
	double **_train_inputs, **_train_outputs;
	double **_test_inputs, **_test_outputs;

	size_t train_data_length, input_length, output_length, test_data_length;
	void DeleteTrainParams();
	void DeleteTestParams();
public:
	DataSet();
	DataSet(size_t train_data_length, size_t input_length, size_t output_length);
	DataSet(size_t train_data_length, size_t test_data_length, size_t input_length, size_t output_length);
	void SetTrainDataParams(size_t train_data_length, size_t input_length, size_t output_length);
	void SetTestDataParams(size_t test_data_length, size_t input_length, size_t output_length);
	void PrintInfo() const;
	~DataSet();
};

class addit {
public:
	double SigmoidFunction(double x);
	void SoftMaxFunction(double* layer, size_t len);
	int _strcpy(char* target, const char* source, int st);
	int _strcpy(char* target, long long num, int st);
	char* GetTimeFromMilliseconds(long long millisecond);
	void plot(double* arr, size_t size);
	template<class T> void Shuffle(T** v1, T** v2, size_t len);
	void LoadX(const char* sourcePath, int len, int slen, double** X);
	void LoadY(const char* sourcePath, int len, int slen, double** Y);
	double to_double(const char* str);
	void printString(const char* str, bool new_line = false);
};

class NeuralNetwork
{
private:
	addit functions;
	size_t layers_count, threads_count;
	size_t* neurons_per_layer;
	double* temp;
	double** layers;
	double** glayers;
	double*** weights;
	double*** gradients;
	double*** moment1;
	double*** moment2;
	ActivationFunction act, llact;
	LossFunction loss;
	Optimizer opt;
	double* errors;
	double betta1, betta2, betta1toTpower, betta2toTpower, alpha_t;

	void NeuralMultiplication(double* input, size_t fln_size);
	void Activation(size_t layer, ActivationFunction);
	void ActDerivative(size_t layer, ActivationFunction);
	void BackProp(double* y, size_t y_size, bool cfl = false);	
	void Optimizing(double, double);
	double* GetLastLayer();
public:
	NeuralNetwork();
	~NeuralNetwork();
	
	void PrintLayers(size_t);
	void PrintWeights();
	void PrintGradients(const char*, size_t); 
	void PrintInfo(); 

	void LoadWeights(const char*);
	void SaveWeights(const char*);

	int Init(vector<size_t> npl, ActivationFunction act, ActivationFunction llact, LossFunction loss, Optimizer opt); //
	void Train(double** inputs, double** ys, size_t train_size, size_t input_length, size_t output_length, size_t lvl, size_t batch, double alpha, bool print = false);
	double* Predict(double* input, size_t fln_size);
};