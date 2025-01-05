#include <iostream>
#include "NeuralNetwork.h"

int main() {
	{
	/*NeuralNetwork square;

	square.Init({1,1}, 
		ActivationFunction::Linear,
		ActivationFunction::Linear,
		LossFunction::SquaredError,
		Optimizer::Adam
	);
	
	vector<vector<float>> sInputs;
	// vector<vector<float>> sInputs = { {5}, {2}, {1}, {6}, {8}, {5}, {9}, {1}, {0}, {3} };
	vector<vector<float>> sAnswers;

	for (size_t i = 0; i < 100.; ++i) {
		sInputs.push_back({(float)(i)});
		sAnswers.push_back({ sInputs.back()[0] * 1.8f + 12.0f });
	}

	square.Train(sInputs, sAnswers, 450, 8, 0.1f);

	/*vector<float> sTest = {10};
	square.NeuralMultiplication(sTest);
	square.PrintLayers(0);
	square.PrintWeights();

	/*for (auto& v : sInputs) {
		square.NeuralMultiplication(v);
		std::cout << "\n" << v[0] << " : ";
		square.PrintLayers(1);
	}

	vector<float> sTest = { 125 };
	square.NeuralMultiplication(sTest);
	std::cout << "\n" << sTest[0] << " : ";
	square.PrintLayers(1);

	square.SaveWeights("squareplus.txt");
	return 0;
	NeuralNetwork nn;

	nn.Init({10, 32, 2}, ActivationFunction::Sigmoid, ActivationFunction::SoftMax, LossFunction::CrossEntropy, Optimizer::Adam);
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
	nn.Train(inputs, ys, 50, 8, 0.03f);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	std::cout << "\nFunction execution time: " << duration.count() << " miliseconds" << std::endl;

	//nn.LoadWeights("weights_for_odd_even.txt");
	/*vector<float> test(3, 1);
	vector<float> zeros(7, 0);
	test.insert(test.end(), zeros.begin(), zeros.end());

	for (size_t _i = 0; _i < 9; ++_i) {
		size_t _j = _i + rand() % (10 - _i);
		std::swap(test[_i], test[_j]);
	}

	for (const auto& a : inputs[0])
		std::cout << a << " ";
	std::cout << "\n";
	nn.NeuralMultiplication(inputs[0]);
	nn.PrintLayers(2);*/

	//nn.SaveWeights("weights_for_odd_even.txt");
	}
	/*printf("Start Load\n");
	vector<vector<float>> TrainX(60000, vector<float>(28 * 28));
	vector<vector<float>> TrainY(60000, vector<float>(10, 0));
	vector<vector<float>> TestX(10000, vector<float>(28 * 28));
	vector<vector<float>> TestY(10000, vector<float>(10, 0));

	LoadX(TrainX, "Digits2/trainX.txt");
	LoadY(TrainY, "Digits2/trainY.txt");
	LoadX(TestX, "Digits2/testX.txt");
	LoadY(TestY, "Digits2/testY.txt");*/

	// init
	NeuralNetwork nn;
	if (nn.Init(
			{ 784, 128, 10 },
			ActivationFunction::ReLU,
			ActivationFunction::SoftMax,
			LossFunction::CrossEntropy,
			Optimizer::Adam
		) == -1) {printf("no init.\n"); return 0;
	}



	return 0;
	/*
	nn.PrintInfo();
	printf("Start Train\n");
	nn.Train(TrainX, TrainY, 2, 16, 0.01);

	float corrects = .0f;
	size_t i = 0, j = 0;
	for (j = 0; j < 10000; ++j) {
		nn.NeuralMultiplication(TestX[j]);
		
		auto ans = nn.GetLastLayer();
		int realNum = -1, guessNum = -1, guessI = -1;

		for (i = 0; i < 10; ++i) {
			if (TestY[j][i] == 1)
				realNum = i;
			if (ans[i] > guessNum) {
				guessNum = (int)ans[i];
				guessI = i;
			}
		}

		if (guessI == realNum)
			++corrects;
	}

	printf("\n%f%", corrects / 100);
	system("pause");

	return 0;*/
}