//#include"Random_Layer.h"
//#define random(x) (rand()%x)
//
//int main(){
//	srand((int)time(0));
//	double start_time = time(NULL);;
//	//样本数目
//	int sample_num = 60000;
//	//单条样本特征数目
//	int factor_num = 64;
//	//q熵索引
//	double q = 2.0;
//	//隐藏层数目
//	int L = 10;
//	vector<vector<int>>trainning_set = vector<vector<int>>(sample_num, vector<int>(factor_num));
//	for (int i = 0; i < trainning_set.size(); i++){
//		for (int j = 0; j < trainning_set[0].size(); j++){
//			trainning_set[i][j] = int(random(100));
//		}
//	}
//	Random_Layer<int> rl(q, L, trainning_set);
//	vector<vector<double>> h = rl.Build_H();
//	return 0;
//}