#include "CImg.h"
#include <ctime>
#include <iostream>
#include "filter.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_functions.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include <math_functions.h>


typedef struct{
	UCHAR *r;
	UCHAR *g;
	UCHAR *b;
}imageData;

__constant__ float cst_ptr [9];

using namespace std;
using namespace cimg_library;
#define PI 3.14

__global__ void gray_kernel(UCHAR *red, UCHAR *green, UCHAR *blue
					 ,UCHAR *gray_image, UINT w, UINT h
					 )
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int index = tx*h+ty;

	if(index< w*h)
		gray_image[index] = round(0.299*((double)red[index])+ 0.587*((double)green[index]) + 0.114*((double)blue[index]));
}
__global__ void filter(imageData scr, imageData des, float *_filter, 
					   UINT w, UINT h, UINT filterSIZE, 
					   float factor, float bias)
{
	int tx = blockDim.x * blockIdx.x + threadIdx.x;
	int ty = blockDim.y * blockIdx.y + threadIdx.y;
	int index = tx*h+ty;
	if(index < w*h)
	{
		UCHAR r = 0, g = 0, b = 0;

		for(int i = 0; i< filterSIZE; i++)
		{
			for(int j = 0; j < filterSIZE ; j++)
			{
				int imgX = (tx - filterSIZE/2 + i + w)% w;
				int imgY = (ty - filterSIZE/2 + i + h)% h;
				r += scr.r[imgX*h + imgY]* _filter[i*filterSIZE + j];
				g += scr.g[imgX*h + imgY]* _filter[i*filterSIZE + j];
				b += scr.b[imgX*h + imgY]* _filter[i*filterSIZE + j];
			}
		}

		des.r[index] = min(max(int(factor * r + bias), 0), 255); 
		des.g[index] = min(max(int(factor * g + bias), 0), 255); 
		des.b[index] = min(max(int(factor * b + bias), 0), 255); 
	}
}
void call_kernel(imageData data,UINT w,UINT h)
{
	//cudaMemcpyToSymbol (cst_ptr, host_ptr, data_size );
	UCHAR *dr, *dg, *db;
	checkCudaErrors(cudaMalloc((void**)&dr, w*h));
	checkCudaErrors(cudaMalloc((void**)&dg, w*h));
	checkCudaErrors(cudaMalloc((void**)&db, w*h));

	checkCudaErrors(cudaMemcpy(dr,data.r,w*h,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dg,data.g,w*h,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(db,data.b,w*h,cudaMemcpyHostToDevice));

	UCHAR *d_result, *h_result;
	checkCudaErrors(cudaMalloc((void**)&d_result, w*h));
	h_result = (UCHAR*)malloc(w*h);

	dim3 numThreadPerBlock(32,32);
	dim3 grid(ceil((float)w/32),ceil((float)h/32));

	printf("numblock %d\n",grid.x);
	printf("numblock %d\n",grid.y);

	gray_kernel<<<grid,numThreadPerBlock>>>(dr,dg,db,d_result,w,h);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(h_result,d_result,w*h,cudaMemcpyDeviceToHost));
	CImg<UCHAR> result(w,h,1,1);
	for(int i=0;i<w;i++){
		for(int j=0;j<h;j++){
			result(i,j,0,0) = h_result[i*h + j];	
		}
	}
	result.display("GrayScale");
}
void call_kernel_filter(imageData data,UINT w,UINT h)
{
	imageData d_Data, d_Result;
	
	float *d_filter;
	//Malloc memory
	checkCudaErrors(cudaMalloc((void**)&d_Data.r, w*h));
	checkCudaErrors(cudaMalloc((void**)&d_Data.g, w*h));
	checkCudaErrors(cudaMalloc((void**)&d_Data.b, w*h));

	checkCudaErrors(cudaMalloc((void**)&d_Result.r, w*h));
	checkCudaErrors(cudaMalloc((void**)&d_Result.g, w*h));
	checkCudaErrors(cudaMalloc((void**)&d_Result.b, w*h));

	checkCudaErrors(cudaMalloc(&d_filter, 9*sizeof(float)));
	//copy data

	checkCudaErrors(cudaMemcpy(d_Data.r,data.r,w*h,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Data.g,data.g,w*h,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_Data.b,data.b,w*h,cudaMemcpyHostToDevice));
	
	checkCudaErrors(cudaMemcpy(d_filter,filter_Blur,9*sizeof(float),cudaMemcpyHostToDevice));

	dim3 block(32,32);
	dim3 grid(ceil((float)w/32),ceil((float)h/32));
	float factor = 2.0;
	float bias = 0.0;

	filter<<<grid,block>>>(d_Data,d_Result,d_filter,w,h,
		3,factor,bias);

	imageData h_Result;
	 h_Result.r = (UCHAR*)malloc(w*h);
	 h_Result.g = (UCHAR*)malloc(w*h);
	 h_Result.b = (UCHAR*)malloc(w*h);

	checkCudaErrors(cudaMemcpy(h_Result.r,d_Result.r,w*h,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_Result.g,d_Result.g,w*h,cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(h_Result.b,d_Result.b,w*h,cudaMemcpyDeviceToHost));

	CImg<UCHAR> result(w,h,1,3);
	for(int i=0;i<w;i++){
		for(int j=0;j<h;j++){
			result(i,j,0,0) = h_Result.r[i*h + j];	
			result(i,j,0,1) = h_Result.g[i*h + j];
			result(i,j,0,2) = h_Result.b[i*h + j];
		}
	}
	result.display("GrayScale");
	
}
void host_filter(imageData data, UINT w, UINT h)
{
	float factor = 1.0, bias = 0.0;
	for(int i = 0; i<)
}
int main() 
{
	CImg<UCHAR> image("hi.jpg");
	int width = image.width();
	int height = image.height();
	int depth = image.depth();
	

	CImg<UCHAR> gray1(width,height,depth,3);
	CImg<UCHAR> gray2(width,height,depth,3);
	//gray2 = image.blur(1.5);

	UCHAR *hr,*hg,*hb;
	UCHAR *result = new UCHAR[width*height];

	imageData h_image;
	h_image.r = new UCHAR[width*height];
	h_image.g = new UCHAR[width*height];
	h_image.b = new UCHAR[width*height];

	printf("Load data\n");

	for(int i=0;i<width;i++){
		for(int j=0;j<height;j++){
			h_image.r[i*height +j] = image(i,j,0,0); // RED
			h_image.g[i*height +j] = image(i,j,0,1); // GREEN
			h_image.b[i*height +j] = image(i,j,0,2); // BLUE
			//printf("%d \n",h_image.r[i*height +j]);
			//Y = 0.299*R + 0.587*G + 0.114*B
			//result[i*height +j] = round(0.299*((double)h_image.r[i*height +j]) + 0.587*((double)h_image.g[i*height +j]) + 0.114*((double)h_image.b[i*height +j]));
			//gray1(i,j,0,0) = (UCHAR)h_image.r[i*height +j];	
			//gray1(i,j,0,1) = (UCHAR)h_image.g[i*height +j];	
			//gray1(i,j,0,2) = (UCHAR)h_image.b[i*height +j]	;
		}
	}
	printf("size of h_image: %d\n",h_image.b[100]);
	getchar();
	clock_t begin = clock();	
	double elapsed_secs_1 = double(clock() - begin);	
	std::cout << "elapsed time:  " << elapsed_secs_1 << " msecs" << std::endl; 
	//call_kernel(h_image,width,height);
	call_kernel_filter(h_image,width,height);
	//gray1.save("gray1.bmp");
	//gray2.save("gray2.bmp");
 
	//show all images

	//(image,gray1).display("original");

  return 0;
}