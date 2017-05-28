#pragma once
# include <iostream>
# include <vector>
#include "Eigen\Dense"


using namespace std;
using namespace Eigen;

class RegressionLayer {
private:
	VectorXd w;
	double alpha;
public:
	RegressionLayer(int input_size, double a);
	VectorXd getWeights();
	VectorXd forwardPropagation(MatrixXd inputs);
	double getLoss(VectorXd outputs, VectorXd target);
	void backwardPropagation(MatrixXd inputs, VectorXd outputs, VectorXd targets);
};