
#include <string>
#include <time.h>
#include "mnist_reader.h"
using namespace mnist;

int main()
{
	clock_t start, end;
	string filename_test = "t10k-images.idx3-ubyte";
	string filename_train = "train-images.idx3-ubyte";
	int number_of_test_images = 10000;
	int number_of_training_images = 60000;
	//int image_size = 28 * 28;
	int n_rows = 28;
	int n_cols = 28;

	//read MNIST image into double vector
	vector<vector<double>> train_data;
	vector<vector<double>> test_data;

	start = clock();
	read_mnist_images(filename_train, train_data, number_of_training_images, n_rows, n_cols);
	end = clock();
	cout << "train data read time: " << (double)(end - start) << endl;
	cout << "size: " << train_data.size() << endl;
	if (train_data.size() > 0) {
		for (auto data : train_data[0]) cout << data << ",";
		cout << endl;
	}
	start = clock();
	read_mnist_images(filename_test, test_data, number_of_test_images, n_rows, n_cols);
	end = clock();
	cout << "test data read time: " << (double)(end - start) << endl;
	cout << "size: " << test_data.size() << endl;
	if (test_data.size() > 0) {
		for (auto data : test_data[0]) cout << data << ",";
		cout << endl;
	}
	return 0;
}
