#include <iostream>
#include <chrono>
#include "NeuralNetwork.h"

int main() {
	const auto processor_count = std::thread::hardware_concurrency();
	std::cout << processor_count << "\n";

	NeuralNetwork nn;

	nn.Init({784, 256, 10}, ActivationFunction::ReLU, ActivationFunction::SoftMax);
	vector<float> input(784, 0.5);

	std::cout << "Start\n";
	auto start = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < 500; ++i)
		nn.NeuralMultiplication(input);
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Function execution time: " << duration.count() << " miliseconds" << std::endl;

	/*nn.PrintLayers();
	nn.PrintWeights();*/

	return 0;
}