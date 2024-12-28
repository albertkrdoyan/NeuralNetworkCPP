#include <iostream>
#include <chrono>
#include "NeuralNetwork.h"

int main() {
	const auto processor_count = std::thread::hardware_concurrency();
	std::cout << processor_count << "\n";

	NeuralNetwork nn;

	nn.Init({3, 5, 3}, ActivationFunction::ReLU, ActivationFunction::SoftMax, LossFunction::CrossEntropy);
	nn.LoadWeights("weights.txt");

	vector<float> input(3, 0.5);
	vector<float> y = {0,1,0};

	std::cout << "Start\n";
	auto start = std::chrono::high_resolution_clock::now();

	nn.NeuralMultiplication(input);
	nn.BackProp(y);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Function execution time: " << duration.count() << " miliseconds" << std::endl;

	nn.PrintLayers(0);
	nn.PrintWeights();

	nn.PrintGradients("ALL", 0);

	return 0;
}