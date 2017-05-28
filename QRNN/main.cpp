# include "RegressionLayer.h"
# include"Random_Layer.h"
# include <iostream>
# include <fstream>
# include <vector>
# include <math.h>
# include <ctime>
# include "mnist_reader.h"
#define Myrandom(x) (rand()%x)

using namespace std;
using namespace mnist;

void linear_test(double q) {
    srand((int)time(0));
    //样本数目
    int sample_num = 60000;
    int batch_num = 100;
    //单条样本特征数目
    int factor_num = 64;
    //q熵索引
    //隐藏层数目
    int L = 64;

    cout << "Data initializing..." << endl;
    vector<double> weights;
    for (int i = 0; i < factor_num; i++) {
        weights.push_back(Myrandom(10));
    }
    
    vector<vector<double>> all_inputs;
    vector<double> all_target;
    vector<vector<vector<double>>> batched_inputs;
    vector<vector<double>> batched_targets;

    // sample_num samples
    for (int i = 0; i < sample_num; i++) {
        vector<double> temp;
        // factor_num data in every sample, data in range (0, 10)
        for (int j = 0; j < factor_num; j++) {
            temp.push_back(double(Myrandom(10)));
        }
        all_inputs.push_back(temp);
        double sum = 0;
        for (int l = 0; l < factor_num; l++) {
            sum += weights[l] * temp[l];
        }
        all_target.push_back(sum);
    }

    VectorXd v_target = VectorXd(all_target.size());
    for (int i = 0; i < all_target.size(); i++) {
        v_target(i) = all_target[i];
    }

    Random_Layer<double> rl(q, L, all_inputs);
    MatrixXd h = rl.Build_H();
    //int step = sample_num/batch_num;
    //// 20 batches
    //for (int k = 0; k < batch_num; k++) {
    //  vector<vector<double>> inputs;
    //  vector<double> targets;
    //  for(int i = 0; i < step;i++){
    //      inputs.push_back(h[k*step+i]);
    //      targets.push_back(all_target[k*step+i]);
    //  }
    //  batched_inputs.push_back(inputs);
    //  batched_targets.push_back(targets);
    //}
    
    
    cout << "Regression Network training start..." << endl;
    RegressionLayer rgl = RegressionLayer(L, 0.15);
    
    // training start
    double loss = -1;
    int max_times = 50;
    int times = 0;
    double last_loss = 1000000.0;
    while ((loss == -1 || loss > 1000.0)) {
            VectorXd outputs = rgl.forwardPropagation(h);
            loss = rgl.getLoss(outputs, v_target);
            //cout << "# batch: " << i << " finished   loss: " << loss << endl;
            cout << " finished   loss: " << loss << endl;
            rgl.backwardPropagation(h, outputs, v_target);
        if (loss >= last_loss) {
            times++;
        }
        if (times > 1000) {
            break;
        }
    }



	//// test start
	//ofstream myfile("linearoutput.txt", ios::out);
	//vector<vector<double>> inputs;
	//vector<double> targets;
	//for (int i = 0; i < 100; i++) {
	//	vector<double> temp;
	//	// 10 data in every sample, data in range (0, 10)
	//	for (int j = 0; j < factor_num; j++) {
	//		temp.push_back(double(Myrandom(10)));
	//	}
	//	inputs.push_back(temp);
	//	double sum = 0;
	//	for (int l = 0; l < factor_num; l++) {
	//		sum += weights[l] * temp[l];
	//	}
	//	targets.push_back(sum);
	//}
	//MatrixXd inputs2 = rl.getSampleH(inputs);
	//VectorXd result = rgl.forwardPropagation(inputs2);
	//for (int i = 0; i < 100; i++) {
	//	myfile << result(i) << endl;
	//}
	//myfile << endl;
	//for (int i = 0; i < 100; i++) {
	//	myfile << targets[i] << endl;
	//}
	//myfile.close();
}


void mnist_test(double q) {
	srand((int)time(0));
	int batch_num = 1;
	//隐藏层数目
	int L = 64;

	vector<vector<double>> train_data;
	vector<double> train_label;

	cout << "read mnist data..." << endl;
	string filename_train = "train-images.idx3-ubyte";
	string filename_train_label = "train-labels.idx1-ubyte";
	int number_of_training_images = 60000;
	int n_rows = 28;
	int n_cols = 28;
	read_mnist_images(filename_train, train_data, number_of_training_images, n_rows, n_cols);
	read_Mnist_Label(filename_train_label, train_label);

	VectorXd v_target = VectorXd(train_label.size());
	for (int i = 0; i < train_data.size(); i++) {
		v_target(i) = train_label[i];
	}

	cout << "random layer start..." << endl;
	Random_Layer<double> rl(q, L, train_data);
	MatrixXd h = rl.Build_H();

	cout << "Regression Network training start..." << endl;
	RegressionLayer rgl = RegressionLayer(L, 0.005);

	// training start
	double loss = -1;
	int times = 0;
	double last_loss = 1000000.0;
	bool is_end = false;
	while ((loss == -1 || loss > 0.5)) {
		for (int i = 0; i < batch_num; i++) {
			int cols = h.cols();
			int begin = i * number_of_training_images / batch_num;
			int length = number_of_training_images / batch_num;
			MatrixXd inputs = h.block(begin, 0, length, cols);
			VectorXd targets = v_target.block(begin, 0, length, 1);
			VectorXd outputs = rgl.forwardPropagation(inputs);
			loss = rgl.getLoss(outputs, targets);
			cout << "# batch: " << i << " finished   loss: " << loss << endl;
			//cout << " finished   loss: " << loss << endl;
			rgl.backwardPropagation(inputs, outputs, targets);
			if (loss >= last_loss) {
				times++;
			}
			if (times > 1000) {
				break;
			}
		}
		if (is_end) {
			break;
		}
	}

	// Test 
	cout << "test begin..." << endl;
	ofstream myfile("mnistoutput.txt", ios::out);
	vector<vector<double>> test_data;
	vector<double> test_label;

	cout << "read test data..." << endl;
	string filename_test = "t10k-images.idx3-ubyte";
	string filename_test_label = "t10k-labels.idx1-ubyte";
	int number_of_test_images = 10000;
	int t_n_rows = 28;
	int t_n_cols = 28;
	read_mnist_images(filename_test, test_data, number_of_test_images, t_n_rows, t_n_cols);
	read_Mnist_Label(filename_test_label, test_label);

	VectorXd t_v_target = VectorXd(test_label.size());
	for (int i = 0; i < test_data.size(); i++) {
		t_v_target(i) = test_label[i];
	}

	MatrixXd t_h = rl.getSampleH(test_data);
	VectorXd result = rgl.forwardPropagation(t_h);
	for (int i = 0; i < 100; i++) {
		myfile << result(i) << endl;
	}
	myfile << endl;
	for (int i = 0; i < 100; i++) {
		myfile << v_target(i) << endl;
	}
	myfile.close();
}

void test() {
	VectorXd v = VectorXd(10);
	for (int i = 0; i < 10; i++) {
		v(i) = i;
	}
	for (int i = 0; i < 10; i++) {
		cout << v(i) << endl;
	}
	VectorXd v2 = v.block(2, 0, 5, 1);
	cout << v2 << endl;
}

int main() {
	//改这个q
    double q = 1.5;
	linear_test(q);
	mnist_test(q);
	//test();
}
