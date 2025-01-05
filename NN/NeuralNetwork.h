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

double SigmoidFunction(double x);
void SoftMaxFunction(double* layer, size_t len);
int _strcpy(char* target, const char* source, int st);
int _strcpy(char* target, long long num, int st);
char* GetTimeFromMilliseconds(long long millisecond);
void plot(vector<double> arr);
template<class T> void Shuffle(T** v1, T** v2, size_t len);
void LoadX(vector<vector<float>>& X, const char* sourcePath);
void LoadY(vector<vector<float>>& Y, const char* sourcePath);
double to_double(const char* str);
void printString(const char* str);

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
	void BackProp(double* y, size_t y_size, bool cfl = false);
	void LoadWeights(const char*);
	void SaveWeights(const char*);
	void Optimizing(double, double);
	void Train(double** trX, double** trY, size_t size1, size_t size2, size_t size3, int, size_t, double);
	vector<double> GetLastLayer();
};