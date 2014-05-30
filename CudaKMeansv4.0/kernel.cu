//#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <ctime>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_functions.h>
#include <device_functions.h>
#include <device_launch_parameters.h>
#include "TestClass.h"
#include "KMean.h"
using namespace std;
//size of data set
#define nObj		1024		
#define nDim		1024	
#define nClus   	1024
#define RANGE       16
/*
	get the next pow of 2
	reference in cuda reduction
	
	*Checked* Lv 3
*/
int nextPowerOfTwo(int n) {
    n--;
    n = n >>  1 | n;
    n = n >>  2 | n;
    n = n >>  4 | n;
    n = n >>  8 | n;
    n = n >> 16 | n;
    return ++n;
}

__host__ __device__ 
float euclid_distance(
					int  numDims,
                    int    numObjs,
                    int    numClusters,
                    float *d_Data,
                    float *d_Cluster,    
                    int    objectId,
                    int    clusterId)
{
    int i = 0;
    float ans = 0.0;

    for (i = 0; i < numDims; i++) 
	{
		ans += (d_Data[i + objectId*numDims] - d_Cluster[clusterId*numDims + i])*
			   (d_Data[i + objectId*numDims] - d_Cluster[clusterId*numDims + i]); //distance
    }
    return(ans);
}

template <int BLOCK_SIZE>
__global__ static void compute_distance_v2(
		int numObj, int numClusters, int numDims,
		float *d_a,			
		float *d_b, 
		float *d_Distance 
		)
{
	{
		/*get threadid, blockid*/
		int tx = threadIdx.x;                       
		int ty = threadIdx.y;

		int bx = blockIdx.x;
		int by = blockIdx.y;

		/*get row, col of result matrix*/
		int ROW = BLOCK_SIZE * bx + tx;
		int COL = BLOCK_SIZE * by + ty;

		int index = ROW*numClusters + COL;
		/*init share memory*/
		__shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

		float sum = 0;
		float tmp = 0;
		int maxLoop  =  numDims/BLOCK_SIZE + 1;
		for(int m = 0 ; m < maxLoop; m++)
		{
			if ((ROW < numObj)&& (m*BLOCK_SIZE + ty)< numDims) 
			{
				sA[tx][ty]  =  d_a[ROW*numDims +(BLOCK_SIZE*m + ty)];
			}
			else sA[tx][ty]  = 0;

			if ((COL < numClusters )&& (m*BLOCK_SIZE + tx) < numDims) 
			{
				sB[tx][ty]  =  d_b[COL*numDims + (m*BLOCK_SIZE + tx)];
			}
			else sB[tx][ty]  = 0;
			 __syncthreads();

			 for(int k = 0 ; k < BLOCK_SIZE ; k++)
			 {
				 tmp = sA[tx][k] - sB[k][ty];
				 sum += tmp*tmp;
			 }
			 __syncthreads();
		}
		if(ROW < numObj && COL < numClusters)
		{
			d_Distance[index] = sum;
		}
	}
}

__global__ void compute_distance(
		int numObj, int numClusters, int numDims,
		float *d_Data,			
		float *d_Cluster, 
		float *d_Distance /*result*/
		)
{
	int objIdx	= blockDim.x * blockIdx.x + threadIdx.x;
	int clusIdx = blockDim.y * blockIdx.y + threadIdx.y;
	if(objIdx < numObj && clusIdx < numClusters)
	{
		d_Distance[objIdx* numClusters + clusIdx] = euclid_distance(numDims,numObj,numClusters,d_Data,d_Cluster,objIdx,clusIdx);
	}

}

__global__
void find_nearest_cluster(int	numDims,
                          int	numObjs,
                          int	numClusters,   
						  float *d_Distance,
                          int	*d_Member,          
                          int	*d_MemberChange)
{
    extern __shared__ char sharedMemory[];
    unsigned char *membershipChanged	= (unsigned char *)sharedMemory;
	 membershipChanged[threadIdx.x] = 0;
	__syncthreads();

    int objectId = blockDim.x * blockIdx.x + threadIdx.x;
    if (objectId < numObjs) 
	{
        int   index		= 0;
		int   i			= 0;
        float dist		= 0;
		float min_dist	= 0.0f;

		min_dist = d_Distance[objectId*numClusters + 0];
        for (i = 1; i<numClusters; i++)
		{
			dist = d_Distance[objectId*numClusters + i];
            if (dist < min_dist) 
			{
                min_dist = dist;
                index    = i;
            }
        }
		
		if (d_Member[objectId] != index) 
		{
            membershipChanged[threadIdx.x]	= 1;
        }
        d_Member[objectId]				= index;
        __syncthreads();

        for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) 
		{
            if (threadIdx.x < s) 
			{
                membershipChanged[threadIdx.x] += membershipChanged[threadIdx.x + s];
            }
            __syncthreads();//  wait the sharedMemory
        }

        if (threadIdx.x == 0) d_MemberChange[blockIdx.x] = membershipChanged[0]; //get the result
        
    }
}
__global__
void compute_change(int *deviceMemberChange,
                   int	size,    //  size of deviceMemberChange
                   int	size2)   //  power of two
{

    extern __shared__ unsigned int shareMemory[];
    shareMemory[threadIdx.x] = (threadIdx.x < size) ? deviceMemberChange[threadIdx.x] : 0;
    __syncthreads();

    for (unsigned int s = size2 / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shareMemory[threadIdx.x] += shareMemory[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) deviceMemberChange[0] = shareMemory[0];
}
__global__
void update_cluster(float* d_Data		, int* d_Member		, float* d_Cluster
                    , const int nCoords	, const int nObjs	, const int nClusters
                    , const int rowPerThread, const int colPerThread)
{
    for (int cIdx = 0; cIdx < colPerThread; ++cIdx)
    {
        int c = cIdx * gridDim.y * blockDim.y + blockIdx.y * blockDim.y + threadIdx.y;
		if (c >= nCoords)
            break;
        for (int rIdx = 0; rIdx < rowPerThread; ++rIdx)
        {
            int r = rIdx * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
			if (r >= nClusters)
                break;

            float sumVal		= 0.0f;
            int clusterCount	= 0;
            for (int i = 0; i < nObjs; ++i)
            {
				if (d_Member[i] == r)
                {
					sumVal += d_Data[i*nCoords + c];
                    clusterCount++;
                }
            }
            if (clusterCount > 0)
				d_Cluster[nCoords*r + c] = sumVal / clusterCount;
        }
    }
}

int kMeans(
        float  *d_Data,			
        int     numDims,			
        int     numObjs,			
        int     numClusters,		
        float   threshold,		
        int     maxLoop,			
        int    *h_Member,
        float  *d_Cluster)
{
    int		 loop = 0;
    float    delta;          /* % of objects change their clusters */

    int		*d_Member;
    int		*d_MemberChange;
	float   *d_Distance;
    
	/*Setup thread 4 find_nearest_cluster */
	unsigned int numThreadsPerClusterBlock		= 128;
	unsigned int numClusterBlocks				=  (unsigned int)ceil((numObjs / (float)numThreadsPerClusterBlock));
	unsigned int clusterBlockSharedDataSize =    numThreadsPerClusterBlock * sizeof(unsigned char) ;
	
	cudaDeviceProp deviceProp;
    int deviceNum;
    cudaGetDevice(&deviceNum);
    cudaGetDeviceProperties(&deviceProp, deviceNum);
    if (clusterBlockSharedDataSize > deviceProp.sharedMemPerBlock) {
		printf("%d , %d\n",clusterBlockSharedDataSize,deviceProp.sharedMemPerBlock);
        printf("WARNING: CUDA hardware has insufficient block shared memory.\n");
		exit(1);
	}
	/* Setup thread 4 compute_change */
    unsigned int numReductionThreads			= nextPowerOfTwo(numClusterBlocks);
    unsigned int reductionBlockSharedDataSize	= numReductionThreads * sizeof(unsigned int);
	checkCudaErrors(cudaMalloc((void**)&d_Member, numObjs*sizeof(int)));
	checkCudaErrors(cudaMalloc((void**)&d_MemberChange, numReductionThreads*sizeof(unsigned int)));
	checkCudaErrors(cudaMalloc((void**)&d_Distance, numObjs*numClusters*sizeof(float)));

	checkCudaErrors(cudaMemcpy(d_Member, h_Member, numObjs*sizeof(int), cudaMemcpyHostToDevice));
    dim3 szGrid, szBlock;
    int rowPerThread , colPerThread;
	rowPerThread = 1;
	colPerThread = 1;
	
	szBlock.x = (numObjs < 32)?numObjs:32;	
	szBlock.y = (numDims<16)?numDims:16;		
	szBlock.z = 1;

	szGrid.x = (numObjs + szBlock.x*rowPerThread - 1) / (szBlock.x*rowPerThread); 
	szGrid.y = (numDims + szBlock.y*colPerThread - 1) / (szBlock.y*colPerThread);	
	szGrid.z = 1;

	/*Setup threads 4 compute_distance*/
	 dim3 szGrid1, szBlock1;
	szBlock1.x = (numObjs < 64)?numObjs:32;		printf("bx %d",szBlock1.x);
	szBlock1.y = (numClusters < 16)?numClusters:16;		printf("by %d",szBlock1.y);
	szBlock1.z = 1;

	szGrid1.x = (numObjs + szBlock1.x - 1) / (szBlock1.x);   //printf("gx %d",szGrid.x);
	szGrid1.y = (numClusters + szBlock1.y - 1) / (szBlock1.y);	 //printf("gy %d",szGrid.y);
	szGrid1.z = 1;
	unsigned int compute_distance_sharedMemSize = numClusters * numDims * sizeof(float);
	//getchar();
	/*Setup threads 4 compute_distance2*/
	const int BLOCK_SIZE = 16;
	dim3 block(BLOCK_SIZE,BLOCK_SIZE);
	unsigned int gridDimx = (unsigned int)ceil((numObjs / (float)BLOCK_SIZE)); printf("Gx %d ",gridDimx);
	unsigned int gridDimy = (unsigned int)ceil((numClusters / (float)BLOCK_SIZE)); printf("Gx %d ",gridDimy);

	dim3 grid(gridDimx,gridDimy);
    do
    {
		loop+=1;
		compute_distance_v2<BLOCK_SIZE><<<grid,block>>>(numObjs,numClusters,numDims,d_Data,d_Cluster,d_Distance);
		//compute_distance<<<szGrid1,szBlock1>>>(nObj,nClus,nDim,d_Data,d_Cluster,d_Distance);
        //checkCudaErrors(cudaDeviceSynchronize());
        //checkCudaErrors(cudaGetLastError());

        find_nearest_cluster<<< numClusterBlocks, numThreadsPerClusterBlock, clusterBlockSharedDataSize >>>
            (numDims, numObjs, numClusters,
			d_Distance, d_Member, d_MemberChange);

		
        //checkCudaErrors(cudaDeviceSynchronize());
        //checkCudaErrors(cudaGetLastError());

        compute_change <<< 1, numReductionThreads, reductionBlockSharedDataSize >>>
			(d_MemberChange, numClusterBlocks, numReductionThreads);

		
        //checkCudaErrors(cudaDeviceSynchronize());
        //checkCudaErrors(cudaGetLastError());

		update_cluster <<< szGrid, szBlock >>> (
			d_Data, d_Member
			, d_Cluster, numDims, numObjs, numClusters, rowPerThread, colPerThread);


        int d;
		checkCudaErrors(cudaMemcpy(&d, d_MemberChange,sizeof(int), cudaMemcpyDeviceToHost));
        delta = (float)d/numObjs;
		
		//std::cout<<"Test delta"<<delta<<std::endl;
    } 
    while (delta > threshold && loop < maxLoop);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaGetLastError());

	checkCudaErrors(cudaMemcpy(h_Member, d_Member, 
              numObjs*sizeof(int), cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_Member));
	checkCudaErrors(cudaFree(d_MemberChange));
	printf("Loop %d \n",loop);
    return (loop + 1);
}

float* callkMeans(float* h_Data, int nObjs, int nDims, int nCluster, int*& h_Member)
{
    float* d_Data, *d_Cluster, *h_Cluster;

	checkCudaErrors(cudaMalloc((void**)&d_Data, nObjs*nDims*sizeof(float)));						//Malloc d_data
	checkCudaErrors(cudaMalloc((void**)&d_Cluster, nCluster*nDims*sizeof(float)));					//Malloc d_cluster

	checkCudaErrors(cudaMallocHost(&h_Member, nObjs*sizeof(float), cudaHostAllocDefault));
	checkCudaErrors(cudaMallocHost(&h_Cluster, nCluster*nDims*sizeof(float), cudaHostAllocDefault));
	checkCudaErrors(cudaMemcpy(d_Data, h_Data, nObjs*nDims*sizeof(float), cudaMemcpyHostToDevice));//copy h_data to d_data

	//init h_member
	for (int i=0; i<nObjs; i++) 
	{
		if(i<nCluster) h_Member[i] = i;
		else h_Member[i] = -1;
	}
	//init d_cluster
	float* tempCluste = (float*)malloc(sizeof(float)*nCluster*nDims);

	for(int i = 0; i< nCluster; i++)
	{
		for(int j = 0; j< nDims; j++)
		{
			tempCluste[i*nDims + j] = h_Data[i*nDims +j];
		}
	}
	checkCudaErrors(cudaMemcpy(d_Cluster, tempCluste, nCluster*nDims*sizeof(float), cudaMemcpyHostToDevice));//copy h_data to d_data
	free(tempCluste);

    kMeans(d_Data, nDims, nObjs, nCluster, 0, 5000, h_Member, d_Cluster);

    
	checkCudaErrors(cudaMemcpy(h_Cluster, d_Cluster, nCluster*nDims*sizeof(float), cudaMemcpyDeviceToHost));/*copy cluster centroid*/

    //Free device memory
	checkCudaErrors(cudaFree(d_Data));
	checkCudaErrors(cudaFree(d_Cluster));

	return h_Cluster;// return the centroid
}

void test(int nO,int nD, int nC)
{
	if((nO < nC)) 
	{
		printf("Error: numObject < nCluster \n");
		return;
	}
    float	*dataCm = TestClass::initData(nO, nD);								/*init data*/
    int		*members;
    float	*clusters;

	clock_t begin = clock();														/*get time*/
	clusters = callkMeans(dataCm, nO, nD, nC, members);						/*call KMeans CUDA*/
	double elapsed_secs_1 = double(clock() - begin) / CLOCKS_PER_SEC;				/*get time*/
	cout<<nO<<"x"<<nD<<"  Cluster :"<<nC<<endl;
	std::cout << "callkMeans CUDA: " << elapsed_secs_1 << " secs" << std::endl;   


	begin = clock();															
    KMean* cpu_kmeans = new KMean(dataCm,nO,nD,nC,5000);						
    double elapsed_secs_2 = double(clock() - begin) / CLOCKS_PER_SEC;				
    std::cout << "callkMeans CPU: " << elapsed_secs_2 << " secs" << std::endl;

	cout<<"Speed up: "<<elapsed_secs_2/elapsed_secs_1<<endl;
	
	printf("Checking result\n");
	if(TestClass::CheckResult(cpu_kmeans->get_member(),members,nO,nC,nD))
	{
		printf("Pass\n");
	}
	else printf("Fail\n");

	checkCudaErrors(cudaFreeHost(members));
	checkCudaErrors(cudaFreeHost(clusters));
    checkCudaErrors(cudaFreeHost(dataCm));
	checkCudaErrors(cudaDeviceReset());
}


int main(int argc, char** argv)
{
	printf("K-Means");
	srand(time(NULL));		//init for rand baseon curr time
	checkCudaErrors(cudaSetDevice(0));
	UINT no = 0,nd = 0,nc = 0;
	if (checkCmdLineFlag(argc, (const char **)argv, "no"))
    {
        no = getCmdLineArgumentInt(argc, (const char **)argv, "no");
    }
	else
	{
		no = nObj;
	}
	if (checkCmdLineFlag(argc, (const char **)argv, "nd"))
    {
        nd = getCmdLineArgumentInt(argc, (const char **)argv, "nd");
    }
	else
	{
		nd = nDim;
	}

	if (checkCmdLineFlag(argc, (const char **)argv, "nc"))
    {
        nc = getCmdLineArgumentInt(argc, (const char **)argv, "nc");
    }
	else
	{
		nc = nClus;
	}
    test(no,nd,nc);					//Call the test
	printf("Done");
	getchar();
    return 0;
}
