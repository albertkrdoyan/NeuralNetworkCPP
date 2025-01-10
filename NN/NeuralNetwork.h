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

class DataSetP {
private:
	size_t train_data_index = 0, test_data_index = 0;
	void DeleteTrainParams();
	void DeleteTestParams();
protected:
	size_t train_data_length = 0, input_length = 0, output_length = 0, test_data_length = 0;
	double **_train_inputs = nullptr, **_train_outputs = nullptr;
	double **_test_inputs = nullptr, **_test_outputs = nullptr;
public:
	DataSetP();
	DataSetP(size_t train_data_length, size_t input_length, size_t output_length);
	DataSetP(size_t train_data_length, size_t input_length, size_t output_length, size_t test_data_length);
	void SetTrainDataParams(size_t train_data_length, size_t input_length, size_t output_length);
	void SetTestDataParams(size_t test_data_length, size_t input_length, size_t output_length);
	void PrintInfo() const;
	void PrintData();
	void LoadDataFromFile(const char* data_X_path, const char* data_Y_path, const char* tr_tst);
	void AddTrainCase(double* in_case, size_t in_length, double* out_case, size_t out_length);
	void AddTestCase(double* in_case, size_t in_length, double* out_case, size_t out_length);
	~DataSetP();

	friend class NeuralNetworkP;
};

class addit {
public:
	static double SigmoidFunction(double x);
	static void SoftMaxFunction(double* layer, size_t len);
	static int _strcpy(char* target, const char* source, int st);
	static int _strcpy(char* target, long long num, int st);
	static char* GetTimeFromMilliseconds(long long millisecond);
	static void plot(double* arr, size_t size);
	template<class T> static void Shuffle2(T** v1, T** v2, size_t len);
	template<class T> static void Shuffle1(T* v, size_t len);
	static size_t LoadX(const char* sourcePath, size_t len, size_t slen, double** X);
	static size_t LoadY(const char* sourcePath, size_t len, size_t slen, double** Y);
	static double to_double(const char* str);
	static void printString(const char* str, bool new_line = false);
};

class NeuralNetworkP
{
private:
	double betta1, betta2, betta1toTpower, betta2toTpower, alpha_t;
	size_t layers_count, threads_count;
	size_t* neurons_per_layer;
	double* temp;
	double* errors;
	double* dropout_info;
	double** layers;
	double** dropout;
	double** glayers;
	double*** weights;
	double*** gradients;
	double*** moment1;
	double*** moment2;
	ActivationFunction act, llact;
	LossFunction loss;
	Optimizer opt;

	void NeuralMultiplication(double* input, size_t fln_size, bool use_dropout = false);
	void Activation(size_t layer, ActivationFunction, bool use_dropout);
	void ActDerivative(size_t layer, ActivationFunction);
	void BackProp(double* y, size_t y_size, bool cfl = false);	
	void Optimizing(double, double);
	void PrintLayers(size_t);
	void PrintWeights();
	
	void PrintGradients(const char*, size_t);
	double* GetLastLayer();
public:
	NeuralNetworkP();
	~NeuralNetworkP();
	
	void PrintDropout();

	void PrintInfo(); 

	void LoadWeights(const char*);
	void SaveWeights(const char*);

	int Init(vector<size_t> npl, vector<double> dropout, ActivationFunction act, ActivationFunction llact, LossFunction loss, Optimizer opt); //
	void Train(double** inputs, double** ys, size_t train_size, size_t input_length, size_t output_length, size_t lvl, size_t batch, double alpha, bool print = false);
	void Train(DataSetP &ds, size_t lvl, size_t batch, double alpha, bool print = false);
	double* Predict(double* input, size_t fln_size);

	void Test(DataSetP& ds);
	void Save(std::istream& cin);
};