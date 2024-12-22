#include <iostream>
#include <chrono>
#include "NeuralNetwork.h"

int main() {
	NeuralNetwork nn;

	nn.Init({784, 256, 10}, ActivationFunction::ReLU, ActivationFunction::SoftMax);
	vector<float> input(784, 0.5);

	auto start = std::chrono::high_resolution_clock::now();
	/*for (size_t i = 0; i < 2000; ++i)
		nn.NeuralMultiplication(input);*/
	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "Function execution time: " << duration.count() << " miliseconds" << std::endl;

	/*nn.PrintLayers();
	nn.PrintWeights();*/

	return 0;
}