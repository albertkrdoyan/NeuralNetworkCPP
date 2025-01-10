#include <iostream>
#include "NeuralNetwork.h"

int digits();
int digits2();
int CtoF();

int main() {
	// Test yourself
	digits2();
	system("pause");
	return 0;
}

int digits2()
{
	const int tr_len = 60000, tst_len = 10000, img_len = 28 * 28, result_len = 10;

	DataSet digit_rec_dataset(tr_len, img_len, result_len, tst_len);
	digit_rec_dataset.LoadDataFromFile("Digits2\\trainX.txt", "Digits2\\trainY.txt", "Train");
	digit_rec_dataset.LoadDataFromFile("Digits2\\testX.txt", "Digits2\\testY.txt", "Test");

	NeuralNetwork digit_rec_model_perceptron;
	if (digit_rec_model_perceptron.Init(
		{ img_len, 128, result_len }, { 0, 0.25, 0 },
		ActivationFunction::ReLU,
		ActivationFunction::SoftMax,
		LossFunction::CrossEntropy,
		Optimizer::Adam
	) == -1) {
		printf("no init.\n"); return 0;
	}
	digit_rec_model_perceptron.LoadWeights("C:\\Users\\alber\\Desktop\\GitHub\\NeuralNetworkCPP2025\\x64\\Debug\\dabest.txt");

	digit_rec_dataset.PrintInfo();
	digit_rec_model_perceptron.PrintInfo();

	digit_rec_model_perceptron.Train(digit_rec_dataset, 10, 32, 0.0001, true);
	digit_rec_model_perceptron.Test(digit_rec_dataset);
	digit_rec_model_perceptron.Save(std::cin);

	system("plot.py");
	return 0;
}

int digits()
{
	const int tr_len = 60000, tst_len = 10000, img_len = 28 * 28, result_len = 10;

	DataSet digit_rec_dataset(tr_len, img_len, result_len, tst_len);
	digit_rec_dataset.LoadDataFromFile("Digits2\\trainX.txt", "Digits2\\trainY.txt", "Train");
	digit_rec_dataset.LoadDataFromFile("Digits2\\testX.txt", "Digits2\\testY.txt", "Test");

	NeuralNetwork digit_rec_model_perceptron;
	if (digit_rec_model_perceptron.Init(
		{ img_len, 128, result_len }, {0, 0.5, 0},
		ActivationFunction::ReLU,
		ActivationFunction::SoftMax,
		LossFunction::CrossEntropy,
		Optimizer::Adam
	) == -1) {printf("no init.\n"); return 0;}
	//digit_rec_model_perceptron.LoadWeights("Digits2\\digits_784_128_10_adam_0d0001.txt");

	digit_rec_model_perceptron.Train(digit_rec_dataset, 10, 32, 0.01, true);
	digit_rec_model_perceptron.Test(digit_rec_dataset);

	digit_rec_model_perceptron.Save(std::cin);

	system("plot.py");
	return 0;
}

int CtoF()
{
	srand((unsigned int)time(NULL));

	size_t tr_len = 20, tst_len = 10;
	DataSet cels_to_fahr_dataset(tr_len, 1, 1, tst_len);

	double tci[1] = { };
	double tco[1] = { };

	for (size_t i = 0; i < tr_len; ++i) {
		tci[0] = rand() % 50;
		tco[0] = tci[0] * 1.8 + 32;

		cels_to_fahr_dataset.AddTrainCase(tci, 1, tco, 1);
	}

	for (size_t i = 0; i < tst_len; ++i) {
		tci[0] = rand() % 50;
		tco[0] = tci[0] * 1.8 + 32;

		cels_to_fahr_dataset.AddTestCase(tci, 1, tco, 1);
	}

	//cels_to_fahr_dataset.PrintData();

	NeuralNetwork cels_to_fahr_perc_model;
	if (cels_to_fahr_perc_model.Init({ 1, 1 }, {0, 0}, Linear, Linear, SquaredError, Adam) == -1) {
		printf("no init.\n");
		return 0;
	}

	cels_to_fahr_perc_model.Train(cels_to_fahr_dataset, 300, 1, 0.1);
	//cels_to_fahr_perc_model.Test(cels_to_fahr_dataset);

	//cels_to_fahr_perc_model.PrintWeights();

	//system("plot.py");

	cels_to_fahr_perc_model.Save(std::cin);
	return 0;
}