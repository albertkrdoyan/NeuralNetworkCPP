#include <iostream>
#include "NeuralNetwork.h"

int main() {
	srand(time(NULL));

	addit f;

	printf("Loading data\n");
	const int tr_len = 60000, tst_len = 10000, img_len = 28 * 28;
	double** tr_img = new double* [tr_len] {};
	double** tr_img_info = new double* [tr_len] {};
	double** tst_img = new double* [tst_len] {};
	double** tst_img_info = new double* [tst_len] {};

	for (int i = 0; i < tr_len; ++i) {
		tr_img[i] = new double[img_len];
		tr_img_info[i] = new double[10] {0};
	}
	for (int i = 0; i < tst_len; ++i) {
		tst_img[i] = new double[img_len];
		tst_img_info[i] = new double[10] {0};
	}

	f.LoadX("Digits2\\trainX.txt", tr_len, img_len, tr_img);
	f.LoadY("Digits2\\trainY.txt", tr_len, 10, tr_img_info);
	f.LoadX("Digits2\\testX.txt", tst_len, img_len, tst_img);
	f.LoadY("Digits2\\testY.txt", tst_len, 10, tst_img_info);
	
	// init
	NeuralNetwork nn;
	if (nn.Init(
			{ 3, 5, 2 },
			ActivationFunction::ReLU,
			ActivationFunction::SoftMax,
			LossFunction::CrossEntropy,
			Optimizer::Adam
		) == -1) {printf("no init.\n"); return 0;
	}

	printf("Train\n");
	// nn.Train(tr_img, tr_img_info, tr_len, img_len, 10, 5, 32, 0.03);
	nn.LoadWeights("C:\\Users\\alber\\source\\repos\\Neural Network\\weis.dat");

	double* inp = new double[3] {.5, .8, .4};
	double* out = new double[2] {1, 0};

	nn.NeuralMultiplication(inp, 3);
	nn.BackProp(out, 2);

	nn.PrintLayers(0);
	nn.PrintWeights();
	nn.PrintGradients("ALL", 0);

	delete[] inp, out;



	/*double* d = new double[10];
	double count = 0, summ = 0;
	for (int i = 0; i < tst_len; ++i) {
		nn.NeuralMultiplication(tst_img[i], img_len);

		d = nn.GetLastLayer();

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

	std::cout << '\n' << "%% " << 100 * count / tst_len << '\n';*/

	for (int i = 0; i < tr_len; ++i)
		delete[] tr_img[i];
	delete[] tr_img;
	delete[] tr_img_info;

	for (int i = 0; i < tst_len; ++i)
		delete[] tst_img[i];
	delete[] tst_img;
	delete[] tst_img_info;

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