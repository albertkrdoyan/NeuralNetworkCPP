#include <iostream>
#include "NeuralNetwork.h"

int main() {
	NeuralNetwork square;

	square.Init({2,5,1}, ActivationFunction::ReLU, ActivationFunction::ReLU, LossFunction::SquaredError, Optimizer::GradientDescent);

	vector<vector<float>> sInputs = { {5, 1}, {2, 1}, {1, 1}, {6, 1}, {8, 1}, {5, 1}, {9, 1}, {1, 1}, {0, 1}, {3, 1} };
	vector<vector<float>> sAnswers = { {30}, {15}, {10}, {35}, {45}, {30}, {50}, {10}, {5}, {20} };

	square.Train(sInputs, sAnswers, 100, 4, 0.000001f);

	/*vector<float> sTest = {10};
	square.NeuralMultiplication(sTest);
	square.PrintLayers(0);*/
	square.PrintWeights();

	for (auto& v : sInputs) {
		square.NeuralMultiplication(v);
		std::cout << "\n" << v[0] << " : ";
		square.PrintLayers(2);
	}

	vector<float> sTest = { 25, 1 };
	square.NeuralMultiplication(sTest);
	std::cout << "\n" << sTest[0] << " : ";
	square.PrintLayers(2);

	square.SaveWeights("squareplus.txt");
	return 0;
	NeuralNetwork nn;

	nn.Init({20, 32, 2}, ActivationFunction::ReLU, ActivationFunction::SoftMax, LossFunction::CrossEntropy, Optimizer::GradientDescent);
	//nn.LoadWeights("weights.txt");
	nn.LoadWeights("weights_for_odd_even.txt");
	//nn.PrintInfo();
	srand(time(NULL));

	vector<vector<float>> inputs(1000);
	vector<vector<float>> ys(inputs.size());
	
	for (size_t i = 0; i < inputs.size(); ++i) {
		size_t k = rand() % 21;
		inputs[i] = vector<float>(k, 1);
		for (size_t j = k; j < 20; ++j)
			inputs[i].push_back(0);		

		for (size_t _i = 0; _i < 19; ++_i) {
			size_t _j = _i + rand() % (20 - _i);

			std::swap(inputs[i][_i], inputs[i][_j]);
		}

		if (k % 2 == 0)
			ys[i] = { 1, 0 };
		else
			ys[i] = { 0, 1 };
	}

	std::cout << "Start\n";
	auto start = std::chrono::high_resolution_clock::now();
	nn.Train(inputs, ys, 5, 8, 0.0001f);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "\nFunction execution time: " << duration.count() << " miliseconds" << std::endl;

	vector<float> test(15, 1);
	vector<float> zeros(5, 0);
	test.insert(test.end(), zeros.begin(), zeros.end());

	nn.NeuralMultiplication(test);
	nn.PrintLayers(2);

	nn.SaveWeights("weights_for_odd_even.txt");

	/*for (const auto& el : test)
		std::cout << el << " ";*/

	/*nn.PrintLayers(0);
	nn.PrintWeights();

	nn.PrintGradients("ALL", 0);*/

	return 0;
}