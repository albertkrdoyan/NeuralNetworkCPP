#include <iostream>
#include "NeuralNetwork.h"

int main() {
	NeuralNetwork nn;

	nn.Init({28*28, 256, 10}, ActivationFunction::ReLU, ActivationFunction::SoftMax, LossFunction::CrossEntropy, Optimizer::GradientDescent);
	//nn.LoadWeights("weights.txt");
	nn.PrintInfo();

	vector<vector<float>> inputs(60000, vector<float>(28*28, .0f));
	vector<vector<float>> ys(60000, vector<float>(10, .0f));

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