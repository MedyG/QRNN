#include<stdio.h>
#include<iostream>
#include<vector>
#include<math.h>
using namespace std;

//Summary: 计算两个向量之间的距离

//Parameters:

//		a: 一个向量

//		b: 另一个向量

//Return: 返回两个向量之间的距离
template<class T>
double Mat_countDistance(vector<T>& a, vector<T>& b){
	if (a.size() != b.size()){
		cout << "Mat_countDistance()::ERROR::a and b size not match!!" << endl;
		return 0.0;
	}
	double sum = 0;
	for (int i = 0; i < a.size(); i++){
		sum += (a[i] - b[i])*(a[i] - b[i]);
	}
	return pow(sum, 0.5);
}


//Summary: 计算两个向量之间的差

//Parameters:

//		a: 一个向量

//		b: 另一个向量

//Return: 返回两个向量之间的差值
template<class T>
vector<T> Mat_subtract(vector<T>& a, vector<T>& b){
	vector<T> sub(a.size());
	if (a.size() != b.size()){
		cout << "Mat_countDistance()::ERROR::a and b size not match!!" << endl;
		return sub;
	}
	for (int i = 0; i < a.size(); i++){
		sub[i] = a[i]-b[i];
	}
	return sub;
}


//Summary: 计算向量的模

//Parameters:

//		a: 一个向量

//Return: 返回两个向量之间的差值
template<class T>
double Mat_countModulo(vector<T>& a){
	double sum = 0;
	for (int i = 0; i < a.size(); i++){
		sum += a[i]*a[i];
	}
	return pow(sum, 0.5);
}


//Summary: 检查num是否存在于index中(辅助函数)

//Parameters:

//		num: 数值

//		index: index列表

//Return: 返回是否存在的bool值
template<class T>
bool checkExist(T& num, vector<T>& index){
	vector<T>::iterator ret;
	ret = find(index.begin(), index.end(), num);
	if (ret == index.end())
		return false;
	else
		return true;
}

template<class T>
vector<vector<T>> Mat_T(vector<vector<T>>& a){
	vector<vector<T>> mat(a[0].size(),vector<T>(a.size()));
	for (int i = 0; i < a.size();i++)
	for (int j = 0; j < a[0].size(); j++)
		mat[j][i] = a[i][j];
	return mat;
}

// input should be horizontal oriented, one inside vector for one row of a matrix

// precondition: mat, left multiplier, a 2-dimension matrix
//               v, right multiplier, a 1-dimension vector

// postcondition: product of the multiplication, a 1-dimension vector
template<class T>
vector<T> multiply_V(vector<vector<T>> mat, vector<T> v) {
	vector<T> result;
	if (mat.size() == 0) {
		return result;
	}
	if (mat[0].size() != v.size()) {
		return result;
	}
	for (int i = 0; i < mat.size(); i++) {
		double sum = 0;
		for (int j = 0; j < mat[i].size(); j++) {
			sum += mat[i][j] * v[j];
		}
		result.push_back(sum);
	}
	return result;
}