#include <iostream>
#include "NeuralNetwork.h"

int main() {
	//NeuralNetwork square;

	//square.Init({1,1}, 
	//	ActivationFunction::Linear,
	//	ActivationFunction::Linear,
	//	LossFunction::SquaredError,
	//	Optimizer::GradientDescent
	//);

	//vector<vector<float>> sInputs;
	//// vector<vector<float>> sInputs = { {5}, {2}, {1}, {6}, {8}, {5}, {9}, {1}, {0}, {3} };
	//vector<vector<float>> sAnswers;

	//srand(time(NULL));
	//for (size_t i = 0; i < 50; ++i) {
	//	sInputs.push_back({(float)(rand() % 50)});
	//	sAnswers.push_back({sInputs.back()[0] * 1.8f + 2.2f});
	//}

	//square.Train(sInputs, sAnswers, 150, 2, 0.000001f);

	///*vector<float> sTest = {10};
	//square.NeuralMultiplication(sTest);
	//square.PrintLayers(0);*/
	//square.PrintWeights();

	//for (auto& v : sInputs) {
	//	square.NeuralMultiplication(v);
	//	std::cout << "\n" << v[0] << " : ";
	//	square.PrintLayers(1);
	//}

	//vector<float> sTest = { 25 };
	//square.NeuralMultiplication(sTest);
	//std::cout << "\n" << sTest[0] << " : ";
	//square.PrintLayers(1);

	//square.SaveWeights("squareplus.txt");
	//return 0;
	NeuralNetwork nn;

	nn.Init({10, 16, 2}, ActivationFunction::Sigmoid, ActivationFunction::SoftMax, LossFunction::CrossEntropy, Optimizer::GradientDescent);
	srand(time(NULL));

	vector<vector<float>> inputs(1000);
	vector<vector<float>> ys(inputs.size());
	
	for (size_t i = 0; i < inputs.size(); ++i) {
		size_t k = rand() % 11;
		inputs[i] = vector<float>(k, 1);
		for (size_t j = k; j < 10; ++j)
			inputs[i].push_back(0);		

		for (size_t _i = 0; _i < 9; ++_i) {
			size_t _j = _i + rand() % (10 - _i);

			std::swap(inputs[i][_i], inputs[i][_j]);
		}

		if (k % 2 == 0)
			ys[i] = { 1, 0 };
		else
			ys[i] = { 0, 1 };
	}

	std::cout << "Start\n";
	auto start = std::chrono::high_resolution_clock::now();
	nn.Train(inputs, ys, 10, 8, 0.0001f);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "\nFunction execution time: " << duration.count() << " miliseconds" << std::endl;

	//nn.LoadWeights("weights_for_odd_even.txt");
	vector<float> test(2, 1);
	vector<float> zeros(8, 0);
	test.insert(test.end(), zeros.begin(), zeros.end());

	for (size_t _i = 0; _i < 9; ++_i) {
		size_t _j = _i + rand() % (10 - _i);
		std::swap(test[_i], test[_j]);
	}

	nn.NeuralMultiplication(test);
	nn.PrintLayers(2);

	//nn.SaveWeights("weights_for_odd_even.txt");
	return 0;
}