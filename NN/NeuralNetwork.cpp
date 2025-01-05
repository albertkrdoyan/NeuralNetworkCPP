#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork() {
	const auto processor_count = std::thread::hardware_concurrency();
	//printf("Cores in CPU: %u.\n", processor_count);

	this->t = 1;
	this->betta1 = 0.9f;
	this->betta2 = 0.999f;
	this->threads_count = 4;
	this->llact = act = ReLU;
	this->layers_count = 0;
	this->loss = SquaredError;
	this->opt = GradientDescent;
}

NeuralNetwork::~NeuralNetwork()
{
	if (layers_count == 0) return;

	for (size_t i = 0; i < layers_count; ++i) {
		delete[] layers[i];
		delete[] glayers[i];
	}
	delete[] layers;
	delete[] glayers;

	for (size_t i = 0; i < layers_count - 1; ++i) {
		for (size_t j = 0; j < neurons_per_layer[i + 1]; ++j) {
			delete[] weights[i][j];
			delete[] gradients[i][j];
			delete[] moment1[i][j];
			delete[] moment2[i][j];
		}
		delete[] weights[i];
		delete[] gradients[i];
		delete[] moment1[i];
		delete[] moment2[i];
	}
	delete[] weights;
	delete[] gradients;
	delete[] moment1;
	delete[] moment2;
	delete[] neurons_per_layer;
}

int NeuralNetwork::Init(vector<size_t> npl, ActivationFunction act, ActivationFunction llact, LossFunction loss, Optimizer opt) {
	if (npl.size() == 0) return -1;

	layers_count = npl.size();
	neurons_per_layer = new size_t[layers_count];
	for (size_t i = 0; i < layers_count; ++i)
		neurons_per_layer[i] = (int)npl[i];

	layers = new double* [layers_count];
	glayers = new double* [layers_count];

	if (layers == nullptr || glayers == nullptr) return -1;

	for (size_t i = 0; i < layers_count; ++i) {
		layers[i] = new double[neurons_per_layer[i]];
		glayers[i] = new double[neurons_per_layer[i]];

		if (layers[i] == nullptr || glayers[i] == nullptr) return -1;

		for (size_t j = 0; j < neurons_per_layer[i]; ++j)
			glayers[i][j] = layers[i][j] = .0f;
	}

	weights = new double** [layers_count - 1];
	gradients = new double** [layers_count - 1];
	moment1 = new double** [layers_count - 1];
	moment2 = new double** [layers_count - 1];

	if (weights == nullptr || gradients == nullptr || moment1 == nullptr || moment2 == nullptr) return -1;

	std::random_device rd; // Seed for the random number engine
	std::mt19937 gen(rd()); // Mersenne Twister random number engine
	std::uniform_real_distribution<> dist(0.0, 1.0); // Range [0, 1)

	for (size_t i = 0; i < layers_count - 1; ++i) {
		weights[i] = new double* [neurons_per_layer[i + 1]];
		gradients[i] = new double* [neurons_per_layer[i + 1]];
		moment1[i] = new double* [neurons_per_layer[i + 1]];
		moment2[i] = new double* [neurons_per_layer[i + 1]];

		if (weights[i] == nullptr || gradients[i] == nullptr || moment1[i] == nullptr || moment2[i] == nullptr) return -1;

		for (size_t j = 0; j < neurons_per_layer[i + 1]; ++j) {
			weights[i][j] = new double[neurons_per_layer[i] + 1];
			gradients[i][j] = new double[neurons_per_layer[i] + 1];
			moment1[i][j] = new double[neurons_per_layer[i] + 1];
			moment2[i][j] = new double[neurons_per_layer[i] + 1];

			if (weights[i][j] == nullptr || gradients[i][j] == nullptr || moment1[i][j] == nullptr || moment2[i][j] == nullptr) return -1;

			for (size_t k = 0; k < neurons_per_layer[i] + 1; ++k) {
				weights[i][j][k] = static_cast<double>(dist(gen));
				gradients[i][j][k] = moment1[i][j][k] = moment2[i][j][k] = .0f;
			}
		}
	}

	this->act = act;
	this->llact = llact;
	this->loss = loss;
	this->opt = opt;

	return 0;
}

void NeuralNetwork::PrintLayers(size_t layer) {
	for (size_t i = layer; i < (layer == 0 ? layers_count : layer + 1); ++i) {
		printf("Layer [%zu]: ", i);
		for (size_t j = 0; j < neurons_per_layer[i]; ++j)
			printf("%.10f ", layers[i][j]);
		printf("\n");
	}
}

void NeuralNetwork::PrintWeights() {
	for (size_t i = 0; i < layers_count - 1; ++i) {
		printf("Weight [%zu]\n[\n", i);
		for (size_t j = 0; j < neurons_per_layer[i + 1]; ++j) {
			printf("\t[");
			for (size_t k = 0; k < neurons_per_layer[i] + 1; ++k)
				printf(" %.10f", weights[i][j][k]);
			printf(" ]\n");
		}
		printf("]\n");
	}
}

void NeuralNetwork::PrintGradients(const char* printwhat, size_t layer)
{
	if (printwhat == "ALL" || printwhat == "GLAYER") {
		for (size_t i = layer; i < (layer == 0 ? layers_count : layer + 1); ++i) {
			printf("GLayer [%zu]: ", i);
			for (size_t j = 0; j < neurons_per_layer[i]; ++j)
				printf("%.10f ", glayers[i][j]);
			printf("\n");
		}
	}
	if (printwhat == "ALL" || printwhat == "GRAD") {
		for (size_t i = 0; i < layers_count - 1; ++i) {
			printf("Gradients [%zu]\n[\n", i);
			for (size_t j = 0; j < neurons_per_layer[i + 1]; ++j) {
				printf("\t[");
				for (size_t k = 0; k < neurons_per_layer[i] + 1; ++k)
					printf(" %.10f", gradients[i][j][k]);
				printf(" ]\n");
			}
			printf("]\n");
		}
	}
}

void NeuralNetwork::PrintInfo()
{
	printf("Threads count: %zu\nLayers count %zu: ", threads_count, layers_count);
	for (size_t i = 0; i < layers_count; ++i)
		printf("%d, ", neurons_per_layer[i]);
	printf("\nMain activation function: ");
	switch (act)
	{
	case Linear:
		printf("Linear");
		break;
	case ReLU:
		printf("ReLU");
		break;
	case Sigmoid:
		printf("Sigmoid");
		break;
	case SoftMax:
		printf("SoftMax");
		break;
	default:
		break;
	}
	printf("\nLast layer activation function: ");
	switch (llact)
	{
	case Linear:
		printf("Linear");
		break;
	case ReLU:
		printf("ReLU");
		break;
	case Sigmoid:
		printf("Sigmoid");
		break;
	case SoftMax:
		printf("SoftMax");
		break;
	default:
		break;
	}
	printf("\nLoss function: ");
	switch (loss)
	{
	case CrossEntropy:
		printf("Cross Entropy");
		break;
	case SquaredError:
		printf("Squared Error");
		break;
	default:
		break;
	}
	printf("\nOptimization method: ");
	switch (opt)
	{
	case Adam:
		printf("Adam");
		break;
	case GradientDescent:
		printf("Gradient Descent");
		break;
	default:
		break;
	}
	printf("\n");
}

void NeuralNetwork::NeuralMultiplication(double* fln, size_t fln_size) {
	if ((int)fln_size != neurons_per_layer[0]) return;

	size_t i = 0, j = 0, k = 0;
	double sum = 0.f;

	layers[0] = std::ref(fln);

	for (i = 1; i < layers_count; ++i) {
		//#pragma omp parallel for shared(layers, weights, neurons_per_layer) private(j, k)
		for (size_t j = 0; j < neurons_per_layer[i]; ++j) {
			sum = 0.f;

			//#pragma omp simd reduction(+:sum)
			for (size_t k = 0; k < neurons_per_layer[i - 1]; ++k)
				sum += layers[i - 1][k] * weights[i - 1][j][k];

			sum += weights[i - 1][j][neurons_per_layer[i - 1]];
			layers[i][j] = sum;
		}

		Activation(i, (i != layers_count - 1 ? act : llact));
	}
}

void NeuralNetwork::Activation(size_t layer, ActivationFunction act) {
	size_t i = 0;
	switch (act)
	{
	case ReLU:
		for (i = 0; i < neurons_per_layer[layer]; ++i)
			if (layers[layer][i] < 0) layers[layer][i] = 0;
		break;
	case Sigmoid:
		for (i = 0; i < neurons_per_layer[layer]; ++i)
			layers[layer][i] = SigmoidFunction(layers[layer][i]);
		break;
	case SoftMax:
		SoftMaxFunction(layers[layer], neurons_per_layer[layer]);
		break;
	default:
		break;
	}
}

void NeuralNetwork::LoadWeights(const char* path)
{
	size_t n = 0, i = 0, j = 0;

	ifstream WLoad;
	WLoad.open(path);

	if (WLoad.is_open()) {
		char _char = '\0';
		char buff[32]{};
		size_t ind = 0;

		while (true) {
			_char = WLoad.get();

			if (_char == ' ' || _char == -1 || _char == '\r' || _char == '\n') {
				buff[ind] = '\0';
				weights[n][i][j++] = to_double(buff);
				ind = 0;
				if (j == neurons_per_layer[n] + 1) {
					j = 0;
					++i;
				}
				if (i == neurons_per_layer[n + 1]) {
					n++;
					i = 0;
				}
				if (n == layers_count - 1)
					break;
			}
			else
				buff[ind++] = _char;
		}
		WLoad.close();
	}
	else {
		printf("Not Open\n");
	}
}

void NeuralNetwork::SaveWeights(const char* path)
{
	ofstream RSave;
	RSave.open(path);

	if (RSave.is_open()) {
		for (size_t n = 0; n < layers_count - 1; ++n) {
			for (size_t i = 0; i < neurons_per_layer[n + 1]; ++i) {
				for (size_t j = 0; j < neurons_per_layer[n] + 1; ++j) {
					RSave << std::setprecision(10) << weights[n][i][j];
					if (j != neurons_per_layer[n]) RSave << " ";
				}
				RSave << "\n";
			}
		}
		RSave.close();
	}
	else {
		printf("Not Open\n");
	}
}
/*
void NeuralNetwork::BackProp(vector<float>& y, bool calculate_first_layer)
{
	if (y.size() != layers.back().size()) return;
	
	if (loss == LossFunction::CrossEntropy && llact == ActivationFunction::SoftMax) {
		for (size_t i = 0; i < neurons_per_layer.back(); ++i) {
			// dE / dz
			glayers.back()[i] = layers.back()[i] - y[i];
		}
	}
	else {
		for (size_t i = 0; i < neurons_per_layer.back(); ++i) {
			// dE / dz
			if (loss == LossFunction::SquaredError)
				glayers.back()[i] = 2 * (layers.back()[i] - y[i]);

			if (llact == ActivationFunction::ReLU)
				glayers.back()[i] *= (layers.back()[i] > 0 ? 1 : 0);
			else if (llact == ActivationFunction::Sigmoid)
				glayers.back()[i] *= layers.back()[i] * (1 - layers.back()[i]);
		}
	}

	size_t j = 0, i = 0;

	for (int n = layers_count - 2; n >= 0; --n) {
		for (i = 0; i < weights[n].size(); ++i) {
			gradients[n][i].back() += glayers[n + 1][i]; // de/db
			
			for (j = 0; j < weights[n][0].size() - 1; ++j) {
				if (i == 0) glayers[n][j] = .0f;

				gradients[n][i][j] += glayers[n + 1][i] * layers[n][j]; // de/dw
				if (n != 0 || calculate_first_layer)
					glayers[n][j] += glayers[n + 1][i] * weights[n][i][j]; // de/da
			}
		}
		
		if (n != 0 || calculate_first_layer) {
			switch (act) // de/dz(l - 1)
			{
			case ReLU:
				//#pragma omp parallel for
				for (i = 0; n != 0 && i < glayers[n].size(); ++i)
					glayers[n][i] *= (layers[n][i] > 0 ? 1 : 0);
				break;
			case Sigmoid:
				//#pragma omp parallel for
				for (i = 0; n != 0 && i < glayers[n].size(); ++i)
					glayers[n][i] *= layers[n][i] * (1 - layers[n][i]);
				break;
			case SoftMax:
				break;
			default:
				break;
			}
		}
	}
}

void NeuralNetwork::Optimizing(float alpha, float batch)
{
	size_t n = 0, i = 0, j = 0;

	if (opt == GradientDescent) {
		for (n = 0; n < weights.size(); ++n) {
			for (i = 0; i < weights[n].size(); ++i) {
				for (j = 0; j < weights[n][i].size(); ++j) {
					weights[n][i][j] -= alpha * gradients[n][i][j] / batch;
					gradients[n][i][j] = .0f;
				}
			}
		}
	}
	else {
		float m1H = 0, m2H = 0, betta1toTpower = 1.0f, betta2toTpower = 1.0f;
		for (n = 0; n < weights.size(); ++n) {
			for (i = 0; i < weights[n].size(); ++i) {
				for (j = 0; j < weights[n][i].size(); ++j) {
					moment1[n][i][j] = betta1 * moment1[n][i][j] + (1 - betta1) * gradients[n][i][j];
					moment2[n][i][j] = betta2 * moment2[n][i][j] + (1 - betta2) * gradients[n][i][j] * gradients[n][i][j];

					m1H = moment1[n][i][j] / (1 - (betta1toTpower *= betta1));
					m2H = moment2[n][i][j] / (1 - (betta2toTpower *= betta2));

					weights[n][i][j] -= (alpha * m1H) / (sqrt(m2H) + 0.00000001);
					gradients[n][i][j] = .0f;
				}
			}
		}
		t++;
	}

	//ResetGradients();
}

void NeuralNetwork::Train(vector<vector<float>>& inputs, vector<vector<float>>& ys, int lvl, size_t batch, float alpha)
{
	errors.clear();
	size_t size = (inputs.size() / batch), btch = 0, err = 0;
	errors.reserve(size);
	std::chrono::steady_clock::time_point start, end;
	long long duration;

	for (size_t l = 0; l < lvl; ++l) {
		if (l != 0) Shuffle(inputs, ys);
		for (size_t i = 0; i < size; ++i) {//printf("{%zu - %zu}\n", i * batch, (i == size - 1 ? inputs.size() : (i + 1) * batch));
			start = std::chrono::high_resolution_clock::now();
			errors.push_back(.0f);

			//#pragma omp parallel for shared(layers, weights, neurons_per_layer, errors, glayers, gradients) private(btch)
			for (btch = i * batch; btch < (i == size - 1 ? inputs.size() : (i + 1) * batch); ++btch) {
				NeuralMultiplication(inputs[btch]);

				if (loss == SquaredError) {
					for (err = 0; err < ys[btch].size(); ++err) {
						errors.back() += (float)pow(ys[btch][err] - layers.back()[err], 2);
					}
				}
				else if (loss == CrossEntropy) {
					for (err = 0; err < ys[btch].size(); ++err) {
						errors.back() += (ys[btch][err] == 0 ? 0 : -log(layers.back()[err]));
					}
				}

				BackProp(ys[btch]);
			}
			errors.back() /= batch;

			Optimizing(alpha, (float)batch);

			end = std::chrono::high_resolution_clock::now();
			duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
			//printf("Function execution time: %zu miliseconds.", duration);
			if (i % 32 == 0) {
				printf("ETA: ");
				printString(GetTimeFromMilliseconds(((long long)((lvl - l) * size - i - 1)) * duration));
				printf(" --- Batch : %zu/%zu\n", i + l * size, size * lvl);
			}
		}
	}

	SaveWeights("Digits2/ws.txt");

	plot(errors);
}

vector<float> NeuralNetwork::GetLastLayer()
{
	return layers.back();
}
*/