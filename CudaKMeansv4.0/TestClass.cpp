#include "TestClass.h"
#include <iostream>
#include <fstream>
#include <limits>
#include <cuda_runtime.h>
#include <helper_cuda.h>

TestClass::TestClass(void)
{
}
TestClass::~TestClass(void)
{
}
float* TestClass::initData(int m, int n)
{
    float* arr;
    checkCudaErrors(cudaMallocHost(&arr, m*n*sizeof(float), cudaHostAllocDefault));
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
        {
            arr[j*m + i] = i + j + rand()%100/(float)3;
        }
    return arr;
}
void TestClass::TestData(float* f_data, int nObj, int nDim)
{
	for(int i = 0; i< nObj ; i++)
	{
		printf("Data[%d] : ",i);
		for(int j = 0; j< nDim;j++)
		{
			printf("%2.0f ",f_data[i*nDim + j]);
		}
		printf("\n");
	}
}
void TestClass::TestMember(int *i_mem, int nClus)
{
	for(int i = 0 ;i < nClus ; i++)
	{
		printf("Object[%d] in Cluster: %d \n",i,i_mem[i]);
		if(i%10 == 0) printf("\n");
	}
}
bool TestClass::CheckResult(int* CPU_member, int* CUDA_member
						  ,int nObj, int nCluster, int nDim)
{
	bool result = true;
	for(int i = 0; i< nObj ; i++)
	{
		if(CPU_member[i] != CUDA_member[i]) 
		{
			//printf("No.%d: CPU_member = %d  CUDA_member = %d\n",i,CPU_member[i],CUDA_member[i]);
			getchar();
			result =  false;
		}
	}
	return result;
}