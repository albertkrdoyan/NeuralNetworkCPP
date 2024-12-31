#include <iostream>
#include "NeuralNetwork.h"

int main() {
	NeuralNetwork nn;

	nn.Init({50, 128, 2}, ActivationFunction::ReLU, ActivationFunction::SoftMax, LossFunction::CrossEntropy, Optimizer::GradientDescent);
	//nn.LoadWeights("weights.txt");
	nn.PrintInfo();

	vector<vector<float>> inputs(1000, vector<float>(50, .0f));
	vector<vector<float>> ys(1000, vector<float>(2, .0f));

	std::cout << "Start\n";
	auto start = std::chrono::high_resolution_clock::now();

	nn.Train(std::ref(inputs), std::ref(ys), 1, 32, 0.01f);

	auto end = std::chrono::high_resolution_clock::now();

	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "\nFunction execution time: " << duration.count() << " miliseconds" << std::endl;

	/*nn.PrintLayers(0);
	nn.PrintWeights();

	nn.PrintGradients("ALL", 0);*/

	return 0;
}