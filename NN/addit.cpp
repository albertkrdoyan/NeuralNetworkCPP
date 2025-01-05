#include "addit.h"

double SigmoidFunction(double x) {
	return 1 / (1 + exp(-x));
}

void SoftMaxFunction(double* layer, size_t len) {
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

	for (int j = (int)line.size() - 1; j > -1; --j)
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
void Shuffle(vector<T>& v1, vector<T>& v2) {
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
			Y[i++][(size_t)(_c - '0')] = 1.0f;

		read.close();
	}
	else
		printf("File can't be load... filepath<<%s>>\n", sourcePath);
}

double to_double(const char* str) {
	double res = .0f;
	size_t i = 0;

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

	return res + under;
}
void printString(const char* str) {
	for (size_t i = 0; str[i] != '\0'; ++i)
		printf("%c", str[i]);
}