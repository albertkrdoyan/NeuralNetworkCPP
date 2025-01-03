#include "NeuralNetwork.h"

float SigmoidFunction(float);
void SoftMaxFunction(vector<float>&, size_t);
static char* GetTimeFromMilliseconds(long long);

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

int NeuralNetwork::Init(vector<size_t> npl, ActivationFunction act, ActivationFunction llact, LossFunction loss, Optimizer opt) {
	neurons_per_layer = npl;
	layers_count = neurons_per_layer.size();

	layers = vector<vector<float>>(layers_count);
	glayers = vector<vector<float>>(layers_count);

	for (size_t i = 0; i < layers_count; ++i) {
		layers[i] = vector<float>(neurons_per_layer[i], 0);
		glayers[i] = vector<float>(neurons_per_layer[i], 0);
	}

	weights = vector<vector<vector<float>>>(layers_count - 1);
	gradients = vector<vector<vector<float>>>(layers_count - 1);
	moment1 = vector<vector<vector<float>>>(layers_count - 1);
	moment2 = vector<vector<vector<float>>>(layers_count - 1);

	for (size_t i = 0; i < layers_count - 1; ++i) {
		weights[i] = vector<vector<float>>(neurons_per_layer[i + 1], vector<float>(neurons_per_layer[i] + 1, 0.f));
		gradients[i] = vector<vector<float>>(neurons_per_layer[i + 1], vector<float>(neurons_per_layer[i] + 1, 0.f));
		moment1[i] = vector<vector<float>>(neurons_per_layer[i + 1], vector<float>(neurons_per_layer[i] + 1, 0.f));
		moment2[i] = vector<vector<float>>(neurons_per_layer[i + 1], vector<float>(neurons_per_layer[i] + 1, 0.f));
	}

	std::random_device rd; // Seed for the random number engine
	std::mt19937 gen(rd()); // Mersenne Twister random number engine
	std::uniform_real_distribution<> dist(0.0, 1.0); // Range [0, 1)

	for (auto& weight : weights) {
		for (auto& line : weight) {
			for (auto& w : line)
				w = static_cast<float>(dist(gen));
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
		for (const auto& neuron : layers[i])
			printf("%.10f ", neuron);
		printf("\n");
	}
}

void NeuralNetwork::PrintWeights() {
	for (size_t i = 0; i < layers_count - 1; ++i) {
		printf("Weight [%zu]\n[\n", i);
		for (const auto& line : weights[i]) {
			printf("\t[");
			for (const auto& weight : line)
				printf(" %.10f", weight);
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
			for (const auto& neuron : glayers[i])
				printf("%.10f ", neuron);
			printf("\n");
		}
	}
	if (printwhat == "ALL" || printwhat == "GRAD") {
		for (size_t i = 0; i < layers_count - 1; ++i) {
			printf("Gradients [%zu]\n[\n", i);
			for (const auto& line : gradients[i]) {
				printf("\t[");
				for (const auto& weight : line)
					printf(" %.10f", weight);
				printf(" ]\n");
			}
			printf("]\n");
		}
	}
}

void NeuralNetwork::PrintInfo()
{
	printf("Threads count: %zu\nLayers count %zu: ", threads_count, layers_count);
	for (const auto& ns : neurons_per_layer)
		printf("%zu, ", ns);
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

void NeuralNetwork::NeuralMultiplication(vector<float> &fln) {
	if (fln.size() != layers[0].size()) return;

	size_t i = 0, j = 0, k = 0;
	float sum = 0.f;

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
		SoftMaxFunction(layers[layer], layers[layer].size());
		break;
	default:
		break;
	}
}

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

static float to_float(const char* str) {
	float res = .0f;
	size_t i = 0;

	for (; str[i] != '.' && str[i] != '\0'; ++i) {
		res *= 10.0f;
		res += str[i] - '0';
	}
	if (str[i] == '\0')
		return res;
	++i;

	float under = .0f;
	float dec = 1.0f;
	for (; str[i] != '\0'; ++i) {
		dec /= 10.0f;
		under += dec * (str[i] - '0');
	}

	return res + under;
}

void NeuralNetwork::LoadWeights(const char* path)
{
	size_t n = 0, i = 0, j = 0;

	ifstream WLoad;
	WLoad.open(path);

	if (WLoad.is_open()) {
		char _char = '\0';
		char buff[30];
		size_t ind = 0;

		while (true) {
			_char = WLoad.get();

			if (_char == ' ' || _char == -1 || _char == '\r' || _char == '\n') {
				buff[ind] = '\0';
				weights[n][i][j++] = to_float(buff);
				ind = 0;
				if (j == weights[n][i].size()) {
					j = 0;
					++i;
				}
				if (i == weights[n].size()) {
					n++;
					i = 0;
				}
				if (n == weights.size())
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
		for (size_t n = 0; n < weights.size(); ++n) {
			for (size_t i = 0; i < weights[n].size(); ++i) {
				for (size_t j = 0; j < weights[n][i].size(); ++j) {
					RSave << weights[n][i][j];
					if (j != weights[n][i].size() - 1) RSave << " ";
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

void NeuralNetwork::ResetGradients()
{
	for (auto& line : glayers) {
		for (auto& gn : line)
			gn = 0;
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

void printString(const char* str) {
	for (size_t i = 0; str[i] != '\0'; ++i)
		printf("%c", str[i]);
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

float SigmoidFunction(float x) {
	return 1 / (1 + exp(-x));
}

void SoftMaxFunction(vector<float> &layer, size_t len) {
	// softmax function by ChatGPT
	size_t i;
	float m, sum, constant;

	m = -INFINITY;
	for (i = 0; i < len; ++i) {
		if (m < layer[i]) {
			m = layer[i];
		}
	}

	sum = 0.0f;
	for (i = 0; i < len; ++i) {
		sum += exp(layer[i] - m);
	}

	constant = m + log(sum);
	for (i = 0; i < len; ++i) {
		layer[i] = exp(layer[i] - constant);
	}
}

static int _strcpy(char* target, const char* source, int st) {
	int i = 0;

	for (i; source[i] != '\0'; ++i)
		target[st + i] = source[i];

	target[st + i] = '\0';

	return st + i;
}

static int _strcpy(char* target, long long num, int st) {
	int i = 0;

	vector<char> line;

	do {
		line.push_back(num % 10 + '0');
		num -= num % 10;
		num /= 10;
	} while (num != 0);

	for (int j = line.size() - 1; j > -1; --j)
		target[st + i++] = line[j];

	target[st + i] = '\0';

	return st + i;
}

char* GetTimeFromMilliseconds(long long millisecond)
{
	long long seconds = millisecond / 1000;
	millisecond %= 1000;
	long long minutes = seconds / 60;
	seconds %= 60;
	long long hours = minutes / 60;
	minutes %= 60;

	char result[128];
	int i = 0;
	i = _strcpy(result, "Hours: ", i);
	i = _strcpy(result, hours, i);
	i = _strcpy(result, ", minutes: ", i);
	i = _strcpy(result, minutes, i);
	i = _strcpy(result, ", seconds: ", i);
	i = _strcpy(result, seconds, i);
	i = _strcpy(result, ", milliseconds: ", i);
	i = _strcpy(result, millisecond, i);
	return result;
}

void plot(vector<float> arr) {
	ofstream wr;
	wr.open("plot.txt");

	for (size_t i = 0; i < arr.size(); ++i)
		wr << arr[i] << '\n';

	wr.close();

	system("plot.py");
}

template <class T>
void Shuffle(vector<T> &v1, vector<T> &v2) {
	srand(time(NULL));

	for (size_t i = 0; i < v1.size() - 1; ++i) {
		size_t j = i + rand() % (v1.size() - i);

		std::swap(v1[i], v1[j]);
		std::swap(v2[i], v2[j]);
	}
}

void LoadX(vector<vector<float>>& X, const char* sourcePath) {
	ifstream read;
	read.open(sourcePath);
	int i = 0, j = 0;

	if (read.is_open()) {
		char _c = '\0';
		float num[3] = { 0, 0, 0 };
		int _i = 0;

		while (true) {
			_c = read.get();

			if (_c == -1 || _c == ' ') {
				if (num[2] == 0) {
					X[i][j] = num[0] / 255;
					j++;
					if (j == X[i].size()) {
						j = 0;
						i++;
					}
				}
				else {
					while (num[0]-- != 0) {
						X[i][j] = num[1] / 255;
						j++;
						if (j == X[i].size()) {
							j = 0;
							i++;
						}
					}
				}

				if (_c == -1)
					break;

				num[0] = num[1] = num[2] = 0;
				_i = 0;
			}
			else if (_c == ':') {
				_i = 1;
				num[2] = 1;
			}
			else {
				num[_i] *= 10;
				num[_i] += _c - '0';
			}
		}

		read.close();
	}
	else
		printf("File can't be load... filepath<<%s>>\n", sourcePath);
}

void LoadY(vector<vector<float>>& Y, const char* sourcePath) {
	ifstream read;
	read.open(sourcePath);
	int i = 0;

	if (read.is_open()) {
		char _c = '\0';

		while ((_c = read.get()) != -1)
			Y[i++][_c - '0'] = 1.0f;

		read.close();
	}
	else
		printf("File can't be load... filepath<<%s>>\n", sourcePath);
}