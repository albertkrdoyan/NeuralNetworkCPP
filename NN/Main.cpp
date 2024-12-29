#include <iostream>
#include <chrono>
#include "NeuralNetwork.h"

int main() {
	const auto processor_count = std::thread::hardware_concurrency();
	std::cout << processor_count << "\n";

	NeuralNetwork nn;

	nn.Init({28*28, 256, 10}, ActivationFunction::ReLU, ActivationFunction::SoftMax, LossFunction::CrossEntropy);
	//nn.LoadWeights("weights.txt");

	vector<float> input(28*28, 0.5);
	vector<float> y = {0,0,0,0,0,1,0,0,0,0};

	std::cout << "Start\n";
	auto start = std::chrono::high_resolution_clock::now();

	for (size_t i = 0; i < 500; ++i) {
		nn.NeuralMultiplication(input);
		nn.BackProp(y);
		if (i % 16 == 0)
			nn.ResetGradients();
	}

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Function execution time: " << duration.count() << " miliseconds" << std::endl;

	/*nn.PrintLayers(0);
	nn.PrintWeights();

	nn.PrintGradients("ALL", 0);*/

	return 0;
}