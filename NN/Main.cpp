#include <iostream>
#include "NeuralNetwork.h"

int main() {
	NeuralNetwork add;
	add.Init({ 2,2,1 }, ReLU, ReLU, SquaredError, Adam);

	int tr_size = 50;
	double** inputs = new double* [tr_size] {};
	double** y = new double* [tr_size] {};

	srand((unsigned int)time(0));

	for (int i = 0; i < tr_size; ++i) {
		double d1 = rand() % 10, d2 = rand() % 10;
		inputs[i] = new double[2] {d1, d2};
		y[i] = new double[1] {d1 + d2};
	}

	add.Train(inputs, y, tr_size, 2, 1, 1000, 10, 0.01);
	add.SaveWeights("addition_weights_2_2_1.txt");
	system("plot.py");

	double* inp = new double[2]{7.5, 8.05};
	double* ans = add.Predict(inp, 2);

	printf("%f", ans[0]);

	for (int i = 0; i < tr_size; ++i) {
		delete[] inputs[i], y[i];
	}
	delete[] inputs, y;

	return 0;
	srand((unsigned int)time(0));

	addit f;

	printf("Loading data\n");
	auto start = std::chrono::high_resolution_clock::now();
	const int tr_len = 60000, tst_len = 10000, img_len = 28 * 28, res_len = 10;
	double** tr_img = new double* [tr_len] {};
	double** tr_img_info = new double* [tr_len] {};
	double** tst_img = new double* [tst_len] {};
	double** tst_img_info = new double* [tst_len] {};

	for (int i = 0; i < tr_len; ++i) {
		tr_img[i] = new double[img_len];
		tr_img_info[i] = new double[res_len] {0};
	}
	for (int i = 0; i < tst_len; ++i) {
		tst_img[i] = new double[img_len];
		tst_img_info[i] = new double[res_len] {0};
	}

	f.LoadX("Digits2\\trainX.txt", tr_len, img_len, tr_img);
	f.LoadY("Digits2\\trainY.txt", tr_len, res_len, tr_img_info);
	f.LoadX("Digits2\\testX.txt", tst_len, img_len, tst_img);
	f.LoadY("Digits2\\testY.txt", tst_len, res_len, tst_img_info);
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	char* dur = f.GetTimeFromMilliseconds(duration);
	std::cout << "Download compl. : " << dur << "\n";
	delete[] dur;

	// init
	NeuralNetwork nn;
	if (nn.Init(
			{ img_len, 128, res_len },
			ActivationFunction::ReLU,
			ActivationFunction::SoftMax,
			LossFunction::CrossEntropy,
			Optimizer::Adam
		) == -1) {printf("no init.\n"); return 0;
	}

	printf("Train\n");
	start = std::chrono::high_resolution_clock::now();
	//nn.Train(tr_img, tr_img_info, tr_len, img_len, res_len, 20, 32, 0.01);
	end = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	dur = f.GetTimeFromMilliseconds(duration);
	std::cout << "Train compl. : " << dur << "\n";
	delete[] dur;
	//system("plot.py");

	nn.LoadWeights("Digits2\\digits_784_128_10_adam_0d001.txt");

	double* d;
	double count = 0, summ = 0;
	for (int i = 0; i < tst_len; ++i) {
		d = nn.Predict(tst_img[i], img_len);

		int ind = -1, match = -1;
		double max = -1;

		for (int j = 0; j < 10; ++j) {
			if (d[j] > max) {
				max = d[j];
				ind = j;
			}
			if (tst_img_info[i][j] == 1)
				match = j;
		}

		if (match == ind)
			count++;
	}

	printf("\nTest: %.2f%%\n", 100 * count / tst_len);
	///

	count = summ = 0;
	for (int i = 0; i < tr_len; ++i) {
		d = nn.Predict(tr_img[i], img_len);

		int ind = -1, match = -1;
		double max = -1;

		for (int j = 0; j < 10; ++j) {
			if (d[j] > max) {
				max = d[j];
				ind = j;
			}
			if (tr_img_info[i][j] == 1)
				match = j;
		}

		if (match == ind)
			count++;
	}

	printf("\nTrain: %.2f%%\n", 100 * count / tr_len);

	nn.SaveWeights("Digits2\\digits_784_128_10_adam_0d001_c.txt");

	// del
	for (int i = 0; i < tr_len; ++i)
		delete[] tr_img[i];
	delete[] tr_img;
	delete[] tr_img_info;

	for (int i = 0; i < tst_len; ++i)
		delete[] tst_img[i];
	delete[] tst_img;
	delete[] tst_img_info;
	
	system("pause");

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