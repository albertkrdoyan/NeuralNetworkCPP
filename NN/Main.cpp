#include <iostream>
#include "NeuralNetwork.h"

int main() {
	NeuralNetwork nn;
	nn.Init({2, 10, 10, 2});
	nn.PrintLayers();
	nn.PrintWeights();

	return 0;
}