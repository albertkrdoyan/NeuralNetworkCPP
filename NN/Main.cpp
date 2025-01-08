#include <iostream>
#include "NeuralNetwork.h"

int main() {	
	const int tr_len = 60000, tst_len = 10000, img_len = 28 * 28, result_len = 10;

	//auto start = std::chrono::high_resolution_clock::now();

	printf("Loading data\n");
	DataSet digit_rec_dataset(tr_len, tst_len, img_len, result_len);
	digit_rec_dataset.LoadDataFromFile("Digits2\\trainX.txt", "Digits2\\trainY.txt", "Train");
	digit_rec_dataset.LoadDataFromFile("Digits2\\testX.txt", "Digits2\\testY.txt", "Test");
	printf("Download compl. : \n");

	/*auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	char* dur = addit::GetTimeFromMilliseconds(duration);
	std::cout << "Download compl. : " << dur << "\n";
	delete[] dur;*/

	NeuralNetwork digit_rec_model_perceptron;
	if (digit_rec_model_perceptron.Init(
			{ img_len, 128, result_len },
			ActivationFunction::ReLU,
			ActivationFunction::SoftMax,
			LossFunction::CrossEntropy,
			Optimizer::Adam
		) == -1) {printf("no init.\n"); return 0;
	}
	//digit_rec_model_perceptron.LoadWeights("Digits2\\digits_784_128_10_adam_0d0001.txt");

	digit_rec_model_perceptron.Train(digit_rec_dataset, 10, 32, 0.01, true);
	
	digit_rec_model_perceptron.Test(digit_rec_dataset);

	system("plot.py");
	system("pause");

	return 0;
}

/*NeuralNetwork add;
add.Init({ 2,2,1 }, Linear, Linear, SquaredError, Adam);
add.LoadWeights("addition_weights_2_2_1.txt");

int tr_size = 50;
double** inputs = new double* [tr_size] {};
double** y = new double* [tr_size] {};

srand((unsigned int)time(0));

for (int i = 0; i < tr_size; ++i) {
	double d1 = rand() % 10, d2 = rand() % 10;
	inputs[i] = new double[2] {d1, d2};
	y[i] = new double[1] {d1 + d2};
}

//add.Train(inputs, y, tr_size, 2, 1, 1000, 10, 0.01);
//add.SaveWeights("addition_weights_2_2_1.txt");
//system("plot.py");
add.PrintWeights();

double* inp = new double[2]{7.5, -8.55};
double* ans = add.Predict(inp, 2);

printf("%f", ans[0]);

for (int i = 0; i < tr_size; ++i) {
	delete[] inputs[i], y[i];
}
delete[] inputs, y;

return 0;*/