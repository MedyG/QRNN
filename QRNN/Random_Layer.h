#include<iostream>
#include<vector>
#include<stdio.h>
#include<math.h>
#include"mat.h"
#include<time.h>
#include "Eigen\Dense"
using namespace std;
#define Myrandom(x) (rand()%x)
using namespace Eigen;


// class Random_Layer -- 数据预处理，随机层类
template<class T>
class Random_Layer
{
public:
	Random_Layer(double, int, vector<vector<T>>&);
	MatrixXd Build_H();
	MatrixXd getSampleH(vector<vector<T>>&);
private:
	double countDistance(vector<T>&, vector<vector<T>>&);
	
	double q;
	int L;
	vector<vector<T>> trainning_set;
	vector<vector<T>> a;
	vector<double> b;
};


//Summary:  Random_Layer constructor

//Parameters:

//		q: 熵指标取值

//		L : 隐藏层神经元个数

//		trainning_set: 训练集样本特征	
template<class T>
Random_Layer<T>::Random_Layer(double q, int L, vector<vector<T>>& trainning_set)

{
	this->L = L;
	this->q = q;
	this->trainning_set = trainning_set;
}


//Summary:  计算向量到训练集的平均距离

//Parameters:

//		a: 随机获取而得的样本特征

//		X: 所有样本特征

//Return: 返回a到X里面所有样本的平均距离
template<class T>
double Random_Layer<T>::countDistance(vector<T>& a, vector<vector<T>>&X){
	double sum = 0;
	for (vector<T> x : X){
		sum += Mat_countDistance(a, x);
	}
	return sum / X.size();
}


//Summary:  通过随机获取每个神经元的a和计算得到的b，来计算H矩阵

//Parameters: 无

//Return: 返回H
template<class T>
MatrixXd Random_Layer<T>::Build_H(){
	srand((int)time(0));
	vector<int> index;
	int num = 0;
	double Modulo;
	MatrixXd H = MatrixXd(this->trainning_set.size(), L);
	cout << "random a and b" << endl;
	for (int i = 0; i < this->L; i++){
		num = Myrandom(this->trainning_set.size());
		while (checkExist(num, index))
		{
			num = Myrandom(this->trainning_set.size());
		}
		index.push_back(num);
		a.push_back(this->trainning_set[num]);
		b.push_back(countDistance(this->trainning_set[num], this->trainning_set));
	}
	double temptt;
	cout << "counting H" << endl;
	for (int i = 0; i < this->L; i++){
		for (int j = 0; j < this->trainning_set.size(); j++){
			Modulo = Mat_countModulo(Mat_subtract(this->trainning_set[j], a[i]));
			temptt = pow(1 + (q - 1)*(Modulo * Modulo / (b[i] * b[i])), 1 / (1 - q));
			//cout <<i << " " << j << " " << Modulo << " "   << b[i] << " "<< temptt << endl;
			H(j,i)= temptt;
		}
	}
	return H;
}

//Summary:  通过随机获取每个神经元的a和计算得到的b，来计算H矩阵

//Parameters: 无

//Return: 返回H
template<class T>
MatrixXd Random_Layer<T>::getSampleH(vector<vector<T>>& predict_set){
	vector<int> index;
	int num = 0;
	double Modulo;
	MatrixXd H = MatrixXd(predict_set.size(),L);
	cout << "counting sample H" << endl;
	for (int i = 0; i < this->L; i++){
		for (int j = 0; j < predict_set.size(); j++){
			Modulo = Mat_countModulo(Mat_subtract(predict_set[j], this->a[i]));
			H(j, i) = pow(1 + (this->q - 1)*(Modulo * Modulo / (this->b[i] * this->b[i])), 1 / (1 - this->q));
		}
	}
	return H;
}