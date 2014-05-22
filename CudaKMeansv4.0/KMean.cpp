#include "KMean.h"
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <conio.h>
#include <stdio.h>
#include <array>
#include "TestClass.h"
#include <ctime>
using namespace std;
KMean::KMean(float* data, int nObjs, int nDims, int nClus, int maxLoop):
isChange(true)
{
	n_cluster	= nClus;
	n_dim		= nDims;
	n_object	= nObjs;
	max_loop	= maxLoop;
	run(data);
}
void KMean::setData(float *dat)
{

}

void KMean::run(float * data)
{
	f_data_set = (float*)malloc(n_object*n_dim*sizeof(float));
	for(unsigned int i(0); i<n_object; i++)
	{
		for(int j = 0 ; j< n_dim ; j++)
		{
			f_data_set[i*n_dim + j] = data[i*n_dim + j] ;
		}
	}
	f_centroid = new float[n_cluster*n_dim*sizeof(float)];
	i_member   = new int[n_object*sizeof(int)];
	clock_t begin = clock();
	Process();//
	double elapsed_secs = double(clock() - begin) / CLOCKS_PER_SEC;
	std::cout << "CPU KMeans process time: " << elapsed_secs << " secs" << std::endl;
}
KMean::~KMean(void)
{

}
float KMean::Euclidean(float* f_data, float* f_cluster, int objid, int clusterid)
{
	float k(0.0);
	for(int i = 0 ; i < n_dim ; i++)
	{
		k += (f_data[objid*n_dim + i] - f_cluster[clusterid*n_dim + i])*
			 (f_data[objid*n_dim + i] - f_cluster[clusterid*n_dim + i]);
	}
	return k;
}
void KMean::Process()
{
	if(n_cluster > n_object) 
	{
			cout<<"Error number of groups > elements"<<endl;
			return;	
	}
	else
	{
		int loop = 0;
		isChange = true;
		for(int i = 0 ; i < n_object ; i++)
			i_member[i] = -1;
			//init centroid
		for(unsigned int i(0); i<n_cluster; i++)
		{
			i_member[i] = i;
			for(int j = 0 ; j< n_dim ; j++)
			{
				f_centroid[i*n_dim + j] = f_data_set[i*n_dim + j];
			}
		}
		do
		{
			loop += 1;
			isChange = false;
			for(unsigned int i(0);i<n_object;i++)
			{
				float mindis = Euclidean(f_data_set,f_centroid,i,0);

				float dis(0.0);
				int indx = 0;
				for(int j = 1; j < n_cluster; j++)
				{
					dis = Euclidean(f_data_set,f_centroid,i,j);
					
					if(dis < mindis) 
					{
						mindis = dis;
						indx   = j;
					}
				}
				//printf("\n");
				if(i_member[i] != indx)
				{
					if(!isChange) isChange = true;
				}
				i_member[i] = indx;
			}
			//ReCaculate centroids
			float *sum = new float[n_dim*sizeof(float)];
			for(int i = 0 ; i< n_cluster ; i++)//of f_centroid 
			{
				for(int i = 0; i< n_dim ; i++)
					sum[i] = 0.0;
				int count = 0 ;

				for(int j = 0 ; j< n_object; j++) //of i_member
				{
					if(i_member[j] == i)
					{
						count  += 1;
						for(int k = 0; k< n_dim ; k++)
						{
							sum[k] += f_data_set[j*n_dim + k];
						}
					}
				}
				for(int j = 0; j<n_dim; j++)
					if(count > 0) f_centroid[i*n_dim + j] = (float)(sum[j]/(float)count);
			}
		}while (isChange == true && loop < max_loop);
		printf("Loop %d\n",loop);
	}
}
