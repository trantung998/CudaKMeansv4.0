#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_functions.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <stdio.h>
#include <cstdlib>
#include <ctime>

void matrixGen(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = rand()%108; //so dep
    }
}
/*
-size=x size of matrix
*/
int main(int argc, char *argv[])
{
	srand(time(NULL));
	printf("Matrix Multiply\n");
	//set device default
	checkCudaErrors(cudaSetDevice(0));
	//get size of matrix
	UINT size = 0;
	if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        size = getCmdLineArgumentInt(argc, (const char **)argv, "device");
    }
	else
	{
		size = 2048;
	}
	//create host matrix
	float* h_a,*h_b,*h_c ;
	checkCudaErrors(cudaMallocHost(&h_a, size*size*sizeof(float), cudaHostAllocDefault));
	checkCudaErrors(cudaMallocHost(&h_b, size*size*sizeof(float), cudaHostAllocDefault));
	checkCudaErrors(cudaMallocHost(&h_c, size*size*sizeof(float), cudaHostAllocDefault));
	matrixGen(h_a,size);
	matrixGen(h_b,size);
	//CPU function
	//GPU function

    return 0;
}
