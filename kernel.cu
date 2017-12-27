#include <stdlib.h>
#include <stdio.h>

#define MAX_THREADS 512
#define MAX_BLOCKS 64


unsigned int nextPow2(unsigned int x);

void getNumBlocksAndNumThreads(int n, int &blocks, int &threads);

void kernel(double *srcImage, 
	    double *smpImage,
	    int srcwidth,
	    int srcheight,
	    int smpwidth,
   	    int smpheight,
	    int srcx,
	    int srcy,
	    double *tempsigmaST,
	    double *tempsigmaS,
    	    double &sigmaTT,
	    double *sigmaTS);

__global__ void reduce(double *input, double *output, int n);

double getSigma(double *input1, double *input2, int n);

__global__ void ReduceLine(double *srcImage, double *smpImage, double *sigmaST, double *sigmaS, int srcx, int srcy, int srcwidth,  int smpwidth);
unsigned int nextPow2(unsigned int x) {
	-- x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++ x;
}

void getNumBlocksAndNumThreads(int n, int &blocks, int &threads) {
	threads = n < MAX_THREADS ? nextPow2(n) : MAX_THREADS;
	blocks = (n + threads - 1) / threads;
	return;	
}

/*
In this function, one block calculate one line, block is one dimession and the thread is also one dimession
*/
__global__ void ReduceLine(double *srcImage, double *smpImage, double *sigmaST, double *sigmaS, int srcx, int srcy, int srcwidth,  int smpwidth)  {
	__shared__ double volatile sigmast[MAX_THREADS];
	//__shared__ double sigmas[MAX_THREADS];
	
	unsigned int srcBegin = (srcx + blockIdx.x) * srcwidth + srcy;		// get the begin point in the srcImage.
	unsigned int smpBegin = blockIdx.x * smpwidth;				// get the begin point in the smpImage.
		
	sigmast[threadIdx.x] = 0.0 ;
	//sigmas[threadIdx.x] = 0.0;
	unsigned int offset = blockDim.x;
	for(int i = 0; threadIdx.x + i < smpwidth; i += offset) {
		sigmast[threadIdx.x] += srcImage[srcBegin + threadIdx.x + i] * smpImage[smpBegin + threadIdx.x + i];
	//	sigmas[threadIdx.x] += srcImage[srcBegin + threadIdx.x + i] * srcImage[srcBegin + threadIdx.x + i];
	}
	__syncthreads();
	
	for(unsigned int s = blockDim.x; s > 64; s /= 2) {
		if(threadIdx.x < s / 2) {
			sigmast[threadIdx.x] += sigmast[threadIdx.x + s / 2];
	//		sigmas[threadIdx.x] += sigmas[threadIdx.x + s / 2];	
		}
		__syncthreads();
	}
	if(threadIdx.x < 32) {
		sigmast[threadIdx.x] += sigmast[threadIdx.x + 32];
		sigmast[threadIdx.x] += sigmast[threadIdx.x + 16];
		sigmast[threadIdx.x] += sigmast[threadIdx.x + 8];
		sigmast[threadIdx.x] += sigmast[threadIdx.x + 4];
		sigmast[threadIdx.x] += sigmast[threadIdx.x + 2];
		sigmast[threadIdx.x] += sigmast[threadIdx.x + 1];
	}	
	if(threadIdx.x == 0) {
		sigmaST[blockIdx.x] = sigmast[0];
	//	sigmaS[blockIdx.x] = sigmas[0];
	}
}

__global__ void  reduce(double *input, double *output, int n) {
	__shared__ volatile double scratch[MAX_THREADS];

	scratch[threadIdx.x] = 0.0;
	unsigned int offset = blockDim.x;
	for(unsigned int i = 0; i + threadIdx.x < n; i += offset) {
		scratch[threadIdx.x] += input[threadIdx.x + i];
	}	
	__syncthreads();
	
	for(unsigned int s = blockDim.x; s > 64; s /= 2) {
		if(threadIdx.x < s / 2) {
			scratch[threadIdx.x] += scratch[threadIdx.x + s / 2];
		}
		__syncthreads();
	}
	if(threadIdx.x < 32) {
		scratch[threadIdx.x] += scratch[threadIdx.x + 32];
		scratch[threadIdx.x] += scratch[threadIdx.x + 16];
		scratch[threadIdx.x] += scratch[threadIdx.x + 8];
		scratch[threadIdx.x] += scratch[threadIdx.x + 4];
		scratch[threadIdx.x] += scratch[threadIdx.x + 2];
		scratch[threadIdx.x] += scratch[threadIdx.x + 1];
	}
	if(threadIdx.x == 0) {
		*output = scratch[0];
	}
}

void kernel(double *srcImage, 
	    double *smpImage,
	    int srcwidth,
	    int srcheight,
	    int smpwidth,
   	    int smpheight,
	    int srcx,
	    int srcy,
	    double *tempsigmaST,
	    double *tempsigmaS,
	    double &sigmaS,
	    double *sigmaST) {

	double sigmast = 0.0;
	double sigmas = 0.0;
	
	int blocks = smpheight, threads = MAX_THREADS;
	dim3 blockD(blocks, 1, 1);
	dim3 threadD(threads, 1, 1);
	
	// check the parameter
	
	cudaThreadSynchronize();
	ReduceLine<<<blockD, threadD>>>(srcImage, smpImage, tempsigmaST, tempsigmaS, srcx, srcy, srcwidth, smpwidth);
	cudaThreadSynchronize();
	
		
	blocks = 1;
	threads = nextPow2(smpheight);
	dim3 BlockD(blocks, 1, 1);
	dim3 ThreadD(threads, 1, 1);

	double *getsigmaST = NULL;
//	double *getsigmaS = NULL;
	cudaMalloc(&getsigmaST, sizeof(double));
//	cudaMalloc(&getsigmaS, sizeof(double));
	
	cudaThreadSynchronize();
	reduce<<<BlockD, ThreadD>>>(tempsigmaST, sigmaST + srcx * srcwidth + srcy, smpheight);
//	cudaThreadSynchronize();
//	reduce<<<BlockD, ThreadD>>>(tempsigmaS, getsigmaS, smpheight);
	cudaThreadSynchronize();

//	cudaMemcpy(&sigmas, getsigmaS, sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpy(&sigmast, getsigmaST, sizeof(double), cudaMemcpyDeviceToHost);
//	sigmaS = sigmas;
	//sigmaST = sigmast;
}


