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

void plot(vector<float>); //

template <class T> void Shuffle(vector<T>&, vector<T>&); //

void LoadX(vector<vector<float>>& X, const char* sourcePath); //
void LoadY(vector<vector<float>>& Y, const char* sourcePath); //

double SigmoidFunction(double); //
void SoftMaxFunction(double*, size_t); //

static char* GetTimeFromMilliseconds(long long); //

static int _strcpy(char* target, const char* source, int st); //
static int _strcpy(char* target, long long num, int st); //
double to_double(const char* str); //
void printString(const char* str); //