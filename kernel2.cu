#include <stdio.h>

#define MAX_THREADS 1024

__global__ void Reduce(double *srcImage,
		   double *smpImage,
		   int srcwidth,
		   int srcheight,
		   int smpwidth,
	   	   int smpheight,
		   double *sigmaST) {
	unsigned int srcx = blockIdx.x;
	unsigned int srcy = threadIdx.x;
	for(unsigned int next = 0; srcy + next < srcwidth - smpwidth; next += MAX_THREADS) {

		double sigmast = 0.0;
		for(unsigned int i = 0; i < smpheight; ++ i) {
			for(unsigned int j = 0; j < smpwidth; ++ j) {
				sigmast += srcImage[(srcx + i) * srcwidth + srcy + next + j] * smpImage[i * smpwidth + j];
			}
		}	
		sigmaST[srcx * srcwidth + srcy + next] = sigmast;
	}

}

void kernel2(double *srcImage,
	     double *smpImage,
	     int srcwidth,
	     int srcheight,
	     int smpwidth,
	     int smpheight,
	     double *sigmaST) {
	
	dim3 blockD(srcheight - smpheight, 1, 1);
	dim3 threadD(MAX_THREADS, 1, 1);
	
	Reduce<<<blockD, threadD>>>(srcImage, smpImage, srcwidth, srcheight, smpwidth, smpheight, sigmaST);
	cudaThreadSynchronize();	
}

