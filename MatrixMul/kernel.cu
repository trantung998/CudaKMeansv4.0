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
#include <iostream>
using namespace std;
#define TSIZE 16
void matrixGen(float *data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = rand()%108; //so dep
		//data[i] = 1; //so dep
    }
}
bool compare(float* result, float* result1, UINT nr,UINT nc)
{
	int size = nr*nc;
	for(int i = 0; i< size ; i++)
	{
		if(result[i] != result1[i]) return false;
	}
	return true;
}
void matMulCPU(float *a,float *b, float *c, UINT rA,UINT cA,UINT rB, UINT cB)
{
	float sum = 0;
	for(int i = 0; i< rA; i++)
	{
		for(int j = 0; j< cB ;j++)
		{
			for(int k = 0 ; k < rB ; k++)
			{
				sum += a[i*cA + k]*b[k*cB + j];
			}
			c[i*cB +j] = sum;
			sum = 0;
		}
	}
}
__global__ void matMulGPU_kernel(float *a,float *b, float *c, UINT rA,UINT cA,UINT rB, UINT cB)
{
	__shared__ float sa[TSIZE][TSIZE];
	__shared__ float sb[TSIZE][TSIZE];
	int tx = threadIdx.x;                       
	int ty = threadIdx.y;

	int row = blockIdx.x*TSIZE + threadIdx.x;
	int col = blockIdx.y*TSIZE + threadIdx.y;
	float sum = 0;
	
	UINT maxloop = cA/TSIZE;
	if(cA%TSIZE != 0) maxloop +=1;
	if(cA<TSIZE) maxloop = 1;
	for(int i = 0 ; i<maxloop; i++)
	{
		if((row < rA) && ((i*TSIZE+ty)<cA))
		{
			sa[tx][ty] = a[row*cA + (i*TSIZE+ty)];
		}
		else sa[tx][ty] = 0;
		if((col < cB) && ((i*TSIZE + tx)<rB))
		{
			sb[tx][ty] = b[(i*TSIZE + tx)*cB + col];
		}
		else sb[tx][ty] = 0;
		__syncthreads();

		for(int k = 0; k<TSIZE;k++)
		{
			sum += sa[tx][k] * sb[k][ty];
		}
		__syncthreads();
	}
	if(row<rA && col<cB)
	{
		c[row*cB + col] = sum;
	}
}
__global__ void matMulGPU_kernel_noshare(float *a,float *b, float *c, UINT rA,UINT cA,UINT rB, UINT cB)
{
	int row = blockDim.x * blockIdx.x + threadIdx.x;
	int col = blockDim.y * blockIdx.y + threadIdx.y;
	if(row < rA && col < cB)
	{
		float sum = 0;
		for(int i = 0 ; i < cA ; i++)
		{
			sum += a[row*cA + i]*b[i*cB + col];
		}
		c[row*cB + col] = sum;
	}
}
void matMulGPU(float *a,float *b, float *c, UINT rA,UINT cA,UINT rB, UINT cB)
{
	float *d_a,*d_b,*d_c;
	checkCudaErrors(cudaMalloc((void**)&d_a, rA*cA*sizeof(float)));	
	checkCudaErrors(cudaMalloc((void**)&d_b, rB*cB*sizeof(float)));	
	checkCudaErrors(cudaMalloc((void**)&d_c, rA*cB*sizeof(float)));	

	checkCudaErrors(cudaMemcpy(d_a,a,rA*cA*sizeof(float),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_b,b,rB*cB*sizeof(float),cudaMemcpyHostToDevice));

	dim3 block(TSIZE,TSIZE);
	dim3 grid(ceil((float)rA/TSIZE),ceil((float)cB/TSIZE));
	cout<<"x"<<grid.x<<"y"<<grid.y<<endl;
	matMulGPU_kernel<<<grid,block>>>(d_a,d_b,d_c,rA,cA,rB,cB);
	//matMulGPU_kernel_noshare<<<grid,block>>>(d_a,d_b,d_c,rA,cA,rB,cB); 
	checkCudaErrors(cudaMemcpy(c,d_c,rA*cB*sizeof(float),cudaMemcpyDeviceToHost));
}
void display(float *a, UINT rA,UINT cA){
	for(int i = 0; i < rA; i++)
	{
		for(int j =0; j< cA; j++)
		{
			cout<<a[i*cA + j]<<" "<<endl;	
		}
		cout<<"\n";
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
	UINT rA = 0;
	UINT cA = 0;
	UINT rB = 0;
	UINT cB = 0;
	if (checkCmdLineFlag(argc, (const char **)argv, "ra"))
    {
        rA = getCmdLineArgumentInt(argc, (const char **)argv, "ra");
    }
	else
	{
		rA = 512;
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "ca"))
    {
        cA = getCmdLineArgumentInt(argc, (const char **)argv, "ca");
		rB = cA;
    }
	else
	{
		cA = 512;
		rB = cA;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "cb"))
    {
        cB = getCmdLineArgumentInt(argc, (const char **)argv, "cb");
    }
	else
	{
		cB = 1024;
	}
	cout<<"Matrix A:("<<rA<<"x"<<cA<<")"<<"\nMatrix B:("<<rB<<"x"<<cB<<")"<<endl;
	//create host matrix
	float* h_a,*h_b,*h_c,*h_c1 ;
	checkCudaErrors(cudaMallocHost(&h_a, rA*cA*sizeof(float), cudaHostAllocDefault));
	checkCudaErrors(cudaMallocHost(&h_b, rB*cB*sizeof(float), cudaHostAllocDefault));
	checkCudaErrors(cudaMallocHost(&h_c, rA*cB*sizeof(float), cudaHostAllocDefault));
	checkCudaErrors(cudaMallocHost(&h_c1, rA*cB*sizeof(float), cudaHostAllocDefault));
	matrixGen(h_a,rA*cA);
	matrixGen(h_b,rB*cB);
	//CPU function
	clock_t begin = clock();
	matMulCPU(h_a,h_b,h_c,rA,cA,rB,cB);
	double elapsed_secs_1 = double(clock() - begin) / CLOCKS_PER_SEC;	
	std::cout << "elapsed_secs_1: " << elapsed_secs_1 << " msecs" << std::endl;   
	//GPU function	
	begin = clock();
	matMulGPU(h_a,h_b,h_c1,rA,cA,rB,cB);
	elapsed_secs_1 = double(clock() - begin) / CLOCKS_PER_SEC;	
	std::cout << "elapsed_secs_1 CUDA: " << elapsed_secs_1 << " msecs" << std::endl;   
	//show result
	if(compare(h_c,h_c1,rA,cB)) cout<<"PASS"<<endl;
	else cout<<"FAIL"<<endl;
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	cudaFreeHost(h_c1);
	//getchar();
    return 0;
}
