# include "RegressionLayer.h"
# include <ctime>
# include <vector>
# include <iostream>
# include "mat.h"
using namespace std;

RegressionLayer::RegressionLayer(int input_size, double a) {
	w = VectorXd(input_size);
	srand(time(0));
	for (int i = 0; i < input_size; i++) {
		w(i) = float(rand() % 10) / 10000.0;
	}
	alpha = a;
}

VectorXd RegressionLayer::getWeights() {
	return w;
}

VectorXd RegressionLayer::forwardPropagation(MatrixXd inputs) {
	VectorXd result = inputs * w;
	return result;
}

double RegressionLayer::getLoss(VectorXd outputs, VectorXd targets) {
	double sum = 0;
	if (outputs.size() != targets.size()) {
		return -1;
	}
	for (int i = 0; i < outputs.size(); i++) {
		double temp = outputs(i) - targets(i);
		sum += temp * temp;
	}
	return sum / 2.0 / outputs.size();
}

void RegressionLayer::backwardPropagation(MatrixXd inputs, VectorXd outputs, VectorXd targets) {
	w = w - inputs.transpose() * (outputs - targets)* alpha / double(outputs.size());
}