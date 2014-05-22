#pragma once
#include <vector>
using namespace std;

class KMean
{
public:
	KMean(float*, int, int, int,int);
	void run(float*);
	float	Euclidean(float* f_data, float* d_cluster, int objid, int clusterid);
	void	Process();
	void	setData(float *data);
	int		*get_member(){return i_member;}
	float	*get_cluster(){return f_centroid;}
	~KMean(void);
private:
	int		n_cluster;
	int		n_object;
	int		n_dim;
	int		max_loop;

	float	*f_data_set;
	float	*f_centroid;
	int		*i_member;
	//kiểm soát vòng lặp
	bool isChange;

	
};

