#include "NeuralNetwork.h"

NeuralNetworkP::NeuralNetworkP() {
	const auto processor_count = std::thread::hardware_concurrency();

	temp = nullptr;
	layers = nullptr;
	glayers = nullptr;
	weights = nullptr;
	gradients = nullptr;
	moment1 = nullptr;
	moment2 = nullptr;
	neurons_per_layer = nullptr;
	errors = nullptr;
	dropout = nullptr;
	dropout_info = nullptr;

	this->betta1 = 0.9;
	this->betta2 = 0.999;
	this->betta1toTpower = 1;
	this->betta2toTpower = 1;
	this->alpha_t = 1;
	this->threads_count = 4;
	this->llact = act = ReLU;
	this->layers_count = 0;
	this->loss = SquaredError;
	this->opt = GradientDescent;
}

NeuralNetworkP::~NeuralNetworkP()
{
	if (layers_count == 0) return;

	if (errors != nullptr) 
		delete[] errors;

	for (size_t i = 0; i < layers_count - 1; ++i) {
		for (size_t j = 0; j < neurons_per_layer[i + 1]; ++j)
			delete[] weights[i][j], gradients[i][j], moment1[i][j], moment2[i][j];

		delete[] weights[i], gradients[i], moment1[i], moment2[i];
	}

	delete[] weights, gradients, moment1, moment2, neurons_per_layer;

	for (size_t i = 0; i < layers_count; ++i)
		delete[] layers[i], glayers[i];

	delete[] layers, glayers;
}

int NeuralNetworkP::Init(vector<size_t> npl, vector<double> _dropout, ActivationFunction act, ActivationFunction llact, LossFunction loss, Optimizer opt) {
	if (npl.size() == 0) return -1;

	layers_count = npl.size();
	
	neurons_per_layer = new size_t[layers_count];
	dropout_info = new double[layers_count];

	for (size_t i = 0; i < layers_count; ++i) {
		neurons_per_layer[i] = (int)npl[i];
		dropout_info[i] = _dropout[i];
	}

	layers = new double* [layers_count];
	glayers = new double* [layers_count];
	dropout = new double* [layers_count];

	if (layers == nullptr || glayers == nullptr) return -1;

	for (size_t i = 0; i < layers_count; ++i) {
		layers[i] = new double[neurons_per_layer[i]] {0};
		glayers[i] = new double[neurons_per_layer[i]] {0};
		dropout[i] = new double[neurons_per_layer[i]] {0};

		if (layers[i] == nullptr || glayers[i] == nullptr || dropout[i] == nullptr) return -1;

		for (size_t j = 0; j < (1 - dropout_info[i]) * neurons_per_layer[i]; ++j)
			dropout[i][j] = 1 / (1 - dropout_info[i]);
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
			weights[i][j] = new double[neurons_per_layer[i] + 1] {0};
			gradients[i][j] = new double[neurons_per_layer[i] + 1] {0};
			moment1[i][j] = new double[neurons_per_layer[i] + 1] {0};
			moment2[i][j] = new double[neurons_per_layer[i] + 1] {0};

			if (weights[i][j] == nullptr || gradients[i][j] == nullptr || moment1[i][j] == nullptr || moment2[i][j] == nullptr) return -1;

			for (size_t k = 0; k < neurons_per_layer[i] + 1; ++k)
				weights[i][j][k] = static_cast<double>(dist(gen));
		}
	}

	this->act = act;
	this->llact = llact;
	this->loss = loss;
	this->opt = opt;

	return 0;
}

void NeuralNetworkP::PrintLayers(size_t layer) {
	for (size_t i = layer; i < (layer == 0 ? layers_count : layer + 1); ++i) {
		printf("Layer [%zu]: ", i);
		for (size_t j = 0; j < neurons_per_layer[i]; ++j)
			printf("%.10f ", layers[i][j]);
		printf("\n");
	}
}

void NeuralNetworkP::PrintWeights() {
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

void NeuralNetworkP::PrintDropout()
{
	for (size_t i = 0; i < layers_count; ++i) {
		for (size_t j = 0; j < neurons_per_layer[i]; ++j)
			printf("%.5f ", dropout[i][j]);
		printf("\n");
	}
}

void NeuralNetworkP::PrintGradients(const char* printwhat, size_t layer)
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

void NeuralNetworkP::PrintInfo()
{
	printf("\nThreads count: %zu\nLayers count %zu: ", threads_count, layers_count);
	for (size_t i = 0; i < layers_count; ++i)
		printf("%zu, ", neurons_per_layer[i]);
	printf("\nMain activation function: ");
	if (act == Linear)
		printf("Linear");
	else if (act == ReLU)
		printf("ReLU");
	else if (act == Sigmoid)
		printf("Sigmoid");
	else if (act == SoftMax)
		printf("SoftMax");
	
	printf("\nLast layer activation function: ");
	if (llact == Linear)
		printf("Linear");
	else if (llact == ReLU)
		printf("ReLU");
	else if (llact == Sigmoid)
		printf("Sigmoid");
	else if (llact == SoftMax)
		printf("SoftMax");

	printf("\nLoss function: ");
	if (loss == CrossEntropy)
		printf("Cross Entropy");
	else if (loss == SquaredError)
		printf("Squared Error");

	printf("\nOptimization method: ");
	if (opt == Adam)
		printf("Adam");
	else if (opt == GradientDescent)
		printf("Gradient Descent");
	
	printf("\n");
}

void NeuralNetworkP::NeuralMultiplication(double* fln, size_t fln_size, bool use_dropout) {
	if ((int)fln_size != neurons_per_layer[0]) return;

	size_t i = 0, j = 0, k = 0;

	temp = layers[0];
	layers[0] = fln;

	for (i = 1; i < layers_count; ++i) {
		//#pragma omp parallel for shared(layers, weights, neurons_per_layer) private(j, k)
		for (size_t j = 0; j < neurons_per_layer[i]; ++j) {
			layers[i][j] = 0.f;

			for (size_t k = 0; k < neurons_per_layer[i - 1]; ++k)
				layers[i][j] += layers[i - 1][k] * weights[i - 1][j][k];

			layers[i][j] += weights[i - 1][j][neurons_per_layer[i - 1]];
		}

		Activation(i, (i != layers_count - 1 ? act : llact), use_dropout);
	}
}

void NeuralNetworkP::Activation(size_t layer, ActivationFunction act, bool use_dropout) {	
	if (act == ReLU) {
		for (size_t i = 0; i < neurons_per_layer[layer]; ++i) {
			if (layers[layer][i] < 0) layers[layer][i] = 0;
			if (use_dropout) layers[layer][i] *= dropout[layer][i];
		}
	}
	else if (act == Sigmoid) {
		for (size_t i = 0; i < neurons_per_layer[layer]; ++i) {
			layers[layer][i] = addit::SigmoidFunction(layers[layer][i]);
			if (use_dropout) layers[layer][i] *= dropout[layer][i];
		}
	}
	else if (act == SoftMax) {
		addit::SoftMaxFunction(layers[layer], neurons_per_layer[layer]);
	}
}

void NeuralNetworkP::LoadWeights(const char* path)
{
	size_t n = 0, i = 0, j = 0, ind = 0;

	ifstream WLoad;
	WLoad.open(path);

	if (WLoad.is_open()) {
		char _char = '\0';
		char buff[32]{};

		while (true) {
			_char = WLoad.get();

			if (_char == ' ' || _char == -1 || _char == '\r' || _char == '\n') {
				buff[ind] = '\0';
				weights[n][i][j++] = addit::to_double(buff);
				ind = 0;
				if (j == neurons_per_layer[n] + 1) {
					j = 0;
					++i;
				}
				if (i == neurons_per_layer[n + 1]) {
					n++;
					i = 0;
				}
				if (n == layers_count - 1 || _char == -1)
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

void NeuralNetworkP::SaveWeights(const char* path)
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

void NeuralNetworkP::Optimizing(double alpha, double batch)
{
	size_t n = 0, i = 0, j = 0;

	if (opt == GradientDescent) {
		for (n = 0; n < layers_count - 1; ++n) {
			for (i = 0; i < neurons_per_layer[n + 1]; ++i) {
				for (j = 0; j < neurons_per_layer[n] + 1; ++j) {
					weights[n][i][j] -= alpha * gradients[n][i][j] / batch;
					gradients[n][i][j] = 0;
				}
			}
		}
	}
	else {
		double m1H = 0, m2H = 0;
		betta1toTpower *= betta1;
		betta2toTpower *= betta2;
		alpha_t = alpha * (sqrt(1 - betta2toTpower)) / (1 - betta1toTpower);

		for (n = 0; n < layers_count - 1; ++n) {
			for (i = 0; i < neurons_per_layer[n + 1]; ++i) {
				for (j = 0; j < neurons_per_layer[n] + 1; ++j) {
					gradients[n][i][j] /= batch;

					moment1[n][i][j] = betta1 * moment1[n][i][j] + (1 - betta1) * gradients[n][i][j];
					moment2[n][i][j] = betta2 * moment2[n][i][j] + (1 - betta2) * gradients[n][i][j] * gradients[n][i][j];

					m1H = moment1[n][i][j] / (1 - betta1toTpower);
					m2H = moment2[n][i][j] / (1 - betta2toTpower);

					weights[n][i][j] -= (alpha_t * m1H) / (sqrt(m2H) + 0.00000001);
					gradients[n][i][j] = 0;
				}
			}
		}
	}

	//ResetGradients();
}

void NeuralNetworkP::BackProp(double* y, size_t y_size, bool calculate_first_layer)
{
	if (y_size != neurons_per_layer[layers_count - 1]) return;

	if (loss == LossFunction::CrossEntropy && llact == ActivationFunction::SoftMax) {
		for (size_t i = 0; i < neurons_per_layer[layers_count - 1]; ++i)			
			glayers[layers_count - 1][i] = layers[layers_count - 1][i] - y[i]; // dE / dz
	}
	else {
		for (size_t i = 0; i < neurons_per_layer[layers_count - 1]; ++i) {			
			if (loss == LossFunction::SquaredError)
				glayers[layers_count - 1][i] = 2 * (layers[layers_count - 1][i] - y[i]); // dE / dz
		}
		ActDerivative(layers_count - 1, llact); // dE / da(l - 1)
	}

	size_t j = 0, i = 0;

	for (int n = (int)layers_count - 2; n >= 0; --n) {
		for (j = 0; (n != 0 || calculate_first_layer) && j < neurons_per_layer[n]; ++j)
			glayers[n][j] = 0;
		for (i = 0; i < neurons_per_layer[n + 1]; ++i) {
			for (j = 0; j < neurons_per_layer[n]; ++j) {
				gradients[n][i][j] += glayers[n + 1][i] * layers[n][j]; // de/dw
				if (n != 0 || calculate_first_layer)
					glayers[n][j] += glayers[n + 1][i] * weights[n][i][j]; // de/da
			}
			gradients[n][i][j] += glayers[n + 1][i]; // de/db
		}

		if (n != 0 || calculate_first_layer)
			ActDerivative(n, act);
	}

	layers[0] = temp;
}

void NeuralNetworkP::ActDerivative(size_t layer, ActivationFunction act) {
	size_t i = 0;

	if (act == ReLU) {
		//#pragma omp parallel for
		for (i = 0; i < neurons_per_layer[layer]; ++i) {
			if (dropout[layer][i] == 0)
				glayers[layer][i] = 0;
			else
				glayers[layer][i] *= (layers[layer][i] > 0 ? 1 : 0);
		}
	}
	else if (act == Sigmoid) {
		//#pragma omp parallel for
		for (i = 0; i < neurons_per_layer[layer]; ++i) {
			if (dropout[layer][i] == 0)
				glayers[layer][i] = 0;
			else
				glayers[layer][i] *= layers[layer][i] * (1 - layers[layer][i]);
		}
	}
	else if (act == Linear) {
		//#pragma omp parallel for
		for (i = 0; i < neurons_per_layer[layer]; ++i) {
			if (dropout[layer][i] == 0)
				glayers[layer][i] = 0;
		}
	}
}

void NeuralNetworkP::Train(double** inputs, double** ys, size_t train_size, size_t input_length, size_t output_length, size_t lvl, size_t batch, double alpha, bool print)
{
	printf("\nTraining\n");
	auto start_ = std::chrono::high_resolution_clock::now();

	size_t size = (train_size / batch), btch = 0, err = 0, print_speed = size / 3;
	if (print_speed == 0) print_speed = 1;

	std::chrono::steady_clock::time_point start, end;
	long long duration = 0;

	if (errors != nullptr)
		delete[] errors;

	errors = new double[size * lvl] {0};
	size_t err_index = 0;

	start = std::chrono::high_resolution_clock::now();
	for (size_t l = 0; l < lvl; ++l) {
		if (l != 0) addit::Shuffle2(inputs, ys, train_size);

		for (size_t i = 0; i < size; ++i) {//printf("{%zu - %zu}\n", i * batch, (i == size - 1 ? inputs.size() : (i + 1) * batch));			
			//#pragma omp parallel for shared(layers, weights, neurons_per_layer, errors, glayers, gradients) private(btch)
			
			for (size_t i = 0; i < layers_count; ++i) {
				if (dropout_info[i] != 0)
					addit::Shuffle1(dropout[i], neurons_per_layer[i]);
			}

			for (btch = i * batch; btch < (i == size - 1 ? train_size : (i + 1) * batch); ++btch) {
				NeuralMultiplication(inputs[btch], input_length, true);

				if (loss == SquaredError) {
					for (err = 0; err < output_length; ++err) {
						errors[err_index] += pow(ys[btch][err] - layers[layers_count - 1][err], 2);
					}
				}
				else if (loss == CrossEntropy) {
					for (err = 0; err < output_length; ++err) {
						errors[err_index] += (ys[btch][err] == 0 ? 0 : -log(layers[layers_count - 1][err]));
					}
				}

				BackProp(ys[btch], output_length);
			}
			errors[err_index++] /= batch;

			Optimizing(alpha, (double)batch);

			if (((i + 1) % print_speed == 0 || i == size - 1) && print) {
				end = std::chrono::high_resolution_clock::now();
				duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
				char* dur = addit::GetTimeFromMilliseconds((duration / print_speed) * (size * (lvl - l) - i));
				addit::printString(dur);
				delete[] dur;
				printf(" --- lvl : %zu/%zu, batch : %zu/%zu -- %.2f seconds.\n", l + 1, lvl, i + 1, size, (double)duration / 1000);
				start = std::chrono::high_resolution_clock::now();
			}
		}
	}

	auto end_ = std::chrono::high_resolution_clock::now();
	auto duration_ = std::chrono::duration_cast<std::chrono::milliseconds>(end_ - start_).count();
	char* dur_ = addit::GetTimeFromMilliseconds(duration_);
	printf("--- Train compl. : ");
	addit::printString(dur_, true);
	delete[] dur_;

	addit::plot(errors, err_index);
}

void NeuralNetworkP::Train(DataSetP &ds, size_t lvl, size_t batch, double alpha, bool print)
{
	Train(ds._train_inputs, ds._train_outputs, ds.train_data_length, ds.input_length, ds.output_length, lvl, batch, alpha, print);
}

double* NeuralNetworkP::Predict(double* input, size_t fln_size)
{
	NeuralMultiplication(input, fln_size, false);
	layers[0] = temp;
	return GetLastLayer();
}

void NeuralNetworkP::Test(DataSetP& ds)
{
	double* d;
	double count = 0, summ = 0;

	if (ds.test_data_index != 0) {
		printf("\nTesting...\n");
		
		for (int i = 0; i < ds.test_data_length; ++i) {
			d = Predict(ds._test_inputs[i], ds.input_length);

			int ind = -1, match = -1;
			double max = -1;

			for (int j = 0; j < 10; ++j) {
				if (d[j] > max) {
					max = d[j];
					ind = j;
				}
				if (ds._test_outputs[i][j] == 1)
					match = j;
			}

			if (match == ind)
				count++;
		}

		printf("Test: % .4f%%\n", 100 * count / ds.test_data_length);
	}
	if (ds.train_data_index != 0) {
		count = summ = 0;
		for (int i = 0; i < ds.train_data_length; ++i) {
			d = Predict(ds._train_inputs[i], ds.input_length);

			int ind = -1, match = -1;
			double max = -1;

			for (int j = 0; j < 10; ++j) {
				if (d[j] > max) {
					max = d[j];
					ind = j;
				}
				if (ds._train_outputs[i][j] == 1)
					match = j;
			}

			if (match == ind)
				count++;
		}

		printf("Train: %.4f%%\n", 100 * count / ds.train_data_length);
	}	
}

void NeuralNetworkP::Save(std::istream& cin)
{
	printf("Save weights?: (y/n)");
	char save = '\0';
	cin >> save;

	if (save == 'y') {
		char save_file_name[101];
		printf("Save as(max 100 characters): ");
		cin >> save_file_name;
		SaveWeights(save_file_name);
	}
}

double* NeuralNetworkP::GetLastLayer()
{
	return layers[layers_count - 1];
}

double addit::SigmoidFunction(double x) {
	return 1 / (1 + exp(-x));
}

void addit::SoftMaxFunction(double* layer, size_t len) {
	// softmax function by ChatGPT
	size_t i;
	double m, sum, constant;

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

int addit::_strcpy(char* target, const char* source, int st) {
	int i = 0;

	for (i; source[i] != '\0'; ++i)
		target[st + i] = source[i];

	target[st + i] = '\0';

	return st + i;
}

int addit::_strcpy(char* target, long long num, int st) {
	int i = 0;

	vector<char> line;

	do {
		line.push_back(num % 10 + '0');
		num -= num % 10;
		num /= 10;
	} while (num != 0);

	for (int j = (int)line.size() - 1; j > -1; --j)
		target[st + i++] = line[j];

	target[st + i] = '\0';

	return st + i;
}

char* addit::GetTimeFromMilliseconds(long long millisecond)
{
	long long seconds = millisecond / 1000;
	millisecond %= 1000;
	long long minutes = seconds / 60;
	seconds %= 60;
	long long hours = minutes / 60;
	minutes %= 60;

	char* result = new char[128] {};
	//char result[128];
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

void addit::plot(double* arr, size_t size) {
	ofstream wr;
	wr.open("plot.txt");

	for (size_t i = 0; i < size; ++i)
		wr << arr[i] << '\n';

	wr.close();

	//system("plot.py");
}

template<class T> void addit::Shuffle2(T** v1, T** v2, size_t len) {
	for (size_t i = 0; i < len - 1; ++i) {
		size_t j = i + rand() % (len - i);

		std::swap(v1[i], v1[j]);
		std::swap(v2[i], v2[j]);
	}
}

template<class T>
void addit::Shuffle1(T* v, size_t len)
{
	for (size_t i = 0; i < len - 1; ++i)
		std::swap(v[i], v[i + rand() % (len - i)]);
}

size_t addit::LoadX(const char* sourcePath, size_t len, size_t slen, double** X) {
	ifstream read;
	read.open(sourcePath);
	int i = 0, j = 0;

	if (read.is_open()) {
		char _c = '\0';
		double num[3] = { 0, 0, 0 };
		int _i = 0;

		while (true) {
			_c = read.get();

			if (_c == -1 || _c == ' ') {
				if (num[2] == 0) {
					X[i][j] = num[0] / 255;
					j++;
					if (j == slen) {
						j = 0;
						i++;
					}
				}
				else {
					while (num[0]-- != 0) {
						X[i][j] = num[1] / 255;
						j++;
						if (j == slen) {
							j = 0;
							i++;
						}
					}
				}

				if (_c == -1 || i == len)
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
		return i;
	}
	else
		printf("File can't be load... filepath<<%s>>\n", sourcePath);

	return 0;
}

size_t addit::LoadY(const char* sourcePath, size_t len, size_t slen, double** Y) {
	ifstream read;
	read.open(sourcePath);
	int i = 0;

	if (read.is_open()) {
		char _c = '\0';

		while (i != len && (_c = read.get()) != -1)
			Y[i++][_c - '0'] = 1.0f;
		
		read.close();
		return i;
	}
	else
		printf("File can't be load... filepath<<%s>>\n", sourcePath);

	return 0;
}

double addit::to_double(const char* str) {
	double res = 0, is_pos = 1;
	size_t i = 0;

	if (str[i] == '-') {
		i++;
		is_pos = -1;
	}

	for (; str[i] != '.' && str[i] != '\0'; ++i) {
		res *= 10.0f;
		res += str[i] - '0';
	}
	if (str[i] == '\0')
		return res;
	++i;

	double under = .0f;
	double dec = 1.0f;
	for (; str[i] != '\0'; ++i) {
		dec /= 10.0f;
		under += dec * (str[i] - '0');
	}

	return is_pos * (res + under);
}

void addit::printString(const char* str, bool new_line) {
	for (size_t i = 0; str[i] != '\0'; ++i)
		printf("%c", str[i]);
	if (new_line) printf("\n");
}

void DataSetP::DeleteTrainParams()
{
	if (this->train_data_length == 0) return;

	for (size_t i = 0; i < this->train_data_length; ++i)
		delete[] this->_train_inputs[i], this->_train_outputs[i];

	delete[] this->_train_inputs, this->_train_outputs;

	this->train_data_length = 0;
}

void DataSetP::DeleteTestParams()
{
	if (this->test_data_length == 0) return;

	for (size_t i = 0; i < this->test_data_length; ++i)
		delete[] this->_test_inputs[i], this->_test_outputs[i];

	delete[] this->_test_inputs, this->_test_outputs;

	this->test_data_length = 0;
}

DataSetP::DataSetP(){}

DataSetP::DataSetP(size_t train_data_length, size_t input_length, size_t output_length)
{
	if (this->train_data_length != 0) return;

	SetTrainDataParams(train_data_length, input_length, output_length);
}

DataSetP::DataSetP(size_t train_data_length, size_t input_length, size_t output_length, size_t test_data_length)
{
	if (this->train_data_length != 0) return;

	SetTrainDataParams(train_data_length, input_length, output_length);
	SetTestDataParams(test_data_length, input_length, output_length);
}

void DataSetP::SetTrainDataParams(size_t train_data_length, size_t input_length, size_t output_length)
{
	if (this->input_length != 0 && this->input_length != input_length) return;
	if (this->output_length != 0 && this->output_length != output_length) return;
	if (this->train_data_length != 0) return;

	this->train_data_length = train_data_length;
	this->input_length = input_length;
	this->output_length = output_length;

	this->_train_inputs = new double* [train_data_length];
	this->_train_outputs = new double* [train_data_length];

	for (size_t i = 0; i < train_data_length; ++i) {
		this->_train_inputs[i] = new double[input_length] {0};
		this->_train_outputs[i] = new double[output_length] {0};
	}
}

void DataSetP::SetTestDataParams(size_t test_data_length, size_t input_length, size_t output_length)
{
	if (this->input_length != 0 && this->input_length != input_length) return;
	if (this->output_length != 0 && this->output_length != output_length) return;
	if (this->test_data_length != 0) return;

	this->test_data_length = test_data_length;
	this->input_length = input_length;
	this->output_length = output_length;

	this->_test_inputs = new double* [test_data_length];
	this->_test_outputs = new double* [test_data_length];

	for (size_t i = 0; i < test_data_length; ++i) {
		this->_test_inputs[i] = new double[input_length] {0};
		this->_test_outputs[i] = new double[output_length] {0};
	}
}

void DataSetP::PrintInfo() const
{
	printf("\nTrain Data Length: %zu\nTest Data Length: %zu\nInput Length: %zu\nOutput length: %zu\n", 
		train_data_length, test_data_length, input_length, output_length);
}

void DataSetP::PrintData()
{
	printf("Train Data\n");
	for (size_t i = 0; i < this->train_data_length; ++i) {
		printf("Inputs: ");
		for (size_t j = 0; j < this->input_length; ++j)
			printf("%.10f ", this->_train_inputs[i][j]);
		printf(" Outputs: ");
		for (size_t j = 0; j < this->output_length; ++j)
			printf("%.10f ", this->_train_outputs[i][j]);
		printf("\n");
	}

	printf("Test Data\n");
	for (size_t i = 0; i < this->test_data_length; ++i) {
		printf("Inputs: ");
		for (size_t j = 0; j < this->input_length; ++j)
			printf("%.10f ", this->_test_inputs[i][j]);
		printf(" Outputs: ");
		for (size_t j = 0; j < this->output_length; ++j)
			printf("%.10f ", this->_test_outputs[i][j]);
		printf("\n");
	}
}

void DataSetP::LoadDataFromFile(const char* data_X_path, const char* data_Y_path, const char* tr_tst_info)
{
	auto start = std::chrono::high_resolution_clock::now();

	if (tr_tst_info == "Train") {
		printf("Loading Train data\n");
		train_data_index = addit::LoadX(data_X_path, this->train_data_length, this->input_length, this->_train_inputs);
		if (addit::LoadY(data_Y_path, this->train_data_length, this->output_length, this->_train_outputs) != train_data_index) {
			printf("Can't load <<%s>>", tr_tst_info);
			DeleteTrainParams();
		}
	}
	else if (tr_tst_info == "Test") {
		printf("Loading Test data\n");
		test_data_index = addit::LoadX(data_X_path, this->test_data_length, this->input_length, this->_test_inputs);
		if (addit::LoadY(data_Y_path, this->test_data_length, this->output_length, this->_test_outputs) != test_data_index) {
			printf("Can't load <<%s>>", tr_tst_info);
			DeleteTestParams();
		}
	}
	else {
		printf("DataSet loading failed...\n");
	}
	auto end = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	char* dur = addit::GetTimeFromMilliseconds(duration);
	printf("--- Download process compl. : ");
	addit::printString(dur, true);
	delete[] dur;
}

void DataSetP::AddTrainCase(double* in_case, size_t in_length, double* out_case, size_t out_length)
{
	if (this->train_data_index == this->train_data_length) return;
	if (this->input_length != in_length || this->output_length != out_length) return;

	for (size_t i = 0; i < in_length; ++i)
		this->_train_inputs[this->train_data_index][i] = in_case[i];

	for (size_t i = 0; i < out_length; ++i)
		this->_train_outputs[this->train_data_index][i] = out_case[i];

	++this->train_data_index;
}

void DataSetP::AddTestCase(double* in_case, size_t in_length, double* out_case, size_t out_length)
{
	if (this->test_data_index == this->test_data_length) return;
	if (this->input_length != in_length || this->output_length != out_length) return;

	for (size_t i = 0; i < in_length; ++i)
		this->_test_inputs[this->test_data_index][i] = in_case[i];

	for (size_t i = 0; i < out_length; ++i)
		this->_test_outputs[this->test_data_index][i] = out_case[i];

	++this->test_data_index;
}

DataSetP::~DataSetP()
{
	DeleteTrainParams();
	DeleteTestParams();

	this->input_length = this->output_length = 0;
}	