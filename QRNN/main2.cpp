//# include "RegressionLayer.h"
//# include"Random_Layer.h"
//# include <iostream>
//# include <fstream>
//# include <vector>
//# include <math.h>
//# include <ctime>
//#define random(x) (rand()%x)
//using namespace std;
//
//int main() {
//	srand((int)time(0));
//	//样本数目
//	int sample_num = 60000;
//	int batch_num = 100;
//	//单条样本特征数目
//	int factor_num = 64;
//	//q熵索引
//	double q = 2.5;
//	//隐藏层数目
//	int L = 50;
//
//	cout << "Data initializing..." << endl;
//	vector<double> weights;
//	for (int i = 0; i < factor_num; i++) {
//		weights.push_back(double(i % 10));
//	}
//
//	vector<vector<double>> all_inputs;
//	vector<double> all_target;
//	vector<vector<vector<double>>> batched_inputs;
//	vector<vector<double>> batched_targets;
//
//	// sample_num samples
//	for (int i = 0; i < sample_num; i++) {
//		vector<double> temp;
//		// factor_num data in every sample, data in range (0, 10)
//		for (int j = 0; j < factor_num; j++) {
//			temp.push_back(double(random(10)));
//		}
//		all_inputs.push_back(temp);
//		double sum = 0;
//		for (int l = 0; l < factor_num; l++) {
//			sum += weights[l] * temp[l];
//		}
//		all_target.push_back(sum);
//	}
//
//	//Random_Layer<double> rl(q, L, all_inputs);
//	//vector<vector<double>> h = rl.Build_H();
//	int step = sample_num / batch_num;
//	// 20 batches
//	for (int k = 0; k < batch_num; k++) {
//		vector<vector<double>> inputs;
//		vector<double> targets;
//		for (int i = 0; i < step; i++){
//			inputs.push_back(all_inputs[k*step + i]);
//			targets.push_back(all_target[k*step + i]);
//		}
//		batched_inputs.push_back(inputs);
//		batched_targets.push_back(targets);
//	}
//
//
//
//	cout << "Regression Network training start..." << endl;
//	RegressionLayer rgl = RegressionLayer(factor_num, 0.001);
//
//	// training start
//	double loss = -1;
//	int max_times = 50;
//	int times = 0;
//	while ((loss == -1 || loss > 100.0)) {
//		times++;
//		for (int i = 0; i < batch_num; i++) {
//			vector<vector<double>> inputs = batched_inputs[i];
//			vector<double> targets = batched_targets[i];
//			vector<double> outputs = rgl.forwardPropagation(inputs);
//			loss = rgl.getLoss(outputs, targets);
//			cout << "# batch: " << i << " finished   loss: " << loss << endl;
//			rgl.backwardPropagation(inputs, outputs, targets);
//		}
//	}
//
//	// test start
//	ofstream myfile("output.txt", ios::out);
//	vector<vector<double>> inputs;
//	vector<double> targets;
//	for (int i = 0; i < 100; i++) {
//		vector<double> temp;
//		// 10 data in every sample, data in range (0, 10)
//		for (int j = 0; j < factor_num; j++) {
//			temp.push_back(double(random(10)));
//		}
//		inputs.push_back(temp);
//		double sum = 0;
//		for (int l = 0; l < factor_num; l++) {
//			sum += weights[l] * temp[l];
//		}
//		targets.push_back(sum);
//	}
//	//vector<vector<double>> inputs2 = rl.getSampleH(inputs);
//	vector<double> result = rgl.forwardPropagation(inputs);
//	for (int i = 0; i < 100; i++) {
//		myfile << result[i] << endl;
//	}
//	myfile << endl;
//	for (int i = 0; i < 100; i++) {
//		myfile << targets[i] << endl;
//	}
//	myfile.close();
//}
//
//
//
//
