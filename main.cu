#include <stdio.h>
#include <stdlib.h>
#include "Bmpformat.h"
#include "BmpProcess.h"
#include "BmpProcess.cc"
#include <iostream>
#include <math.h>
#include <time.h>
#include "timer.h"
#include "kernel2.cu"

using namespace std;


int main() {
	
	char *srcFile = "castleS.bmp";
	char *smpFile = "castleT1.bmp";
	
	BYTE *srcImage = NULL;
	BYTE *smpImage = NULL;
	BYTE *srcGray = NULL;
	BYTE *smpGray = NULL;
	
	int srcwidth, srcheight;
	int smpwidth, smpheight;

	int x = -1, y = -1;
	
	//we need a timer

	// read file and get the gray image
	srcImage = readBmp(srcFile, &srcwidth, &srcheight);
	smpImage = readBmp(smpFile, &smpwidth, &smpheight);

	BYTE *srcTempRGB = ConvertBMPToRGBBuffer(srcImage, &srcwidth, &srcheight);
	BYTE *smpTempRGB = ConvertBMPToRGBBuffer(smpImage, &smpwidth, &smpheight);

	srcGray = RGBtoGray(srcTempRGB, &srcwidth, &srcheight);
	smpGray = RGBtoGray(smpTempRGB, &smpwidth, &smpheight);

	if(srcGray == NULL || smpGray == NULL) {
		printf("Cannot get the gray images.\n");
	} else {
		printf("Read images successfully.\n");
		printf("srcWidth = %d\nsrcHeight = %d\nsmpWidth = %d\nsmpHeight = %d\n", srcwidth, srcheight, smpwidth, smpheight);
	}

	// convert BYTE to double
	double *srcimage = new double[srcwidth * srcheight];
	double *smpimage = new double[smpwidth * smpheight];
	
	for(int i = 0; i < srcheight; ++ i) {
		for(int j = 0; j < srcwidth; ++ j) {
			srcimage[i * srcwidth + j] = (double)srcGray[i * srcwidth + j];
		}
	}
	for(int i = 0; i < smpheight; ++ i) {
		for(int j = 0; j < smpwidth; ++ j) {
			smpimage[i * smpwidth + j] = (double)smpGray[i * smpwidth + j];
		}
	}
	printf("Finish the converting\n");
	
	// calculate sigmaT
	double sigmaT = 0.0;
	for(int i = 0; i < smpheight; i ++) {
		for(int j = 0; j < smpwidth; ++j) {
			sigmaT += smpimage[i * smpwidth + j] * smpimage[i * smpwidth + j];
		}
	}
	printf("sigmaT = %lf\n", sigmaT);
	
	// calculate the sum sigmaS
	double *sumSigmaS = new double[srcwidth * srcheight];

	for(int i = 0; i < srcheight; ++ i) {
		for(int j = 0; j < srcwidth; ++ j) {
			if(i == 0 && j == 0) sumSigmaS[i * srcwidth + j] = srcimage[i * srcwidth + j] * srcimage[i * srcwidth + j];
			else if(i == 0 && j > 0) sumSigmaS[i * srcwidth + j] = sumSigmaS[i * srcwidth + j - 1] + srcimage[i * srcwidth + j] * srcimage[i * srcwidth + j];
			else if(i > 0 && j == 0) sumSigmaS[i * srcwidth + j] = sumSigmaS[(i - 1) * srcwidth + j] + srcimage[i * srcwidth + j] * srcimage[i * srcwidth + j];
			else sumSigmaS[i * srcwidth + j] = sumSigmaS[(i - 1) * srcwidth + j] + sumSigmaS[i * srcwidth + j - 1] - sumSigmaS[(i - 1) * srcwidth + j - 1] + srcimage[i * srcwidth + j] * srcimage[i * srcwidth + j]; 
		}
	}
	

	
	// pass the images to GPU
	double *srcimageGPU = NULL;
	double *smpimageGPU = NULL;
	cudaMalloc(&srcimageGPU, srcwidth * srcheight * sizeof(double));
	cudaMalloc(&smpimageGPU, smpwidth * smpheight * sizeof(double));
	
	// get the temp space
	double *sigmaSGPU = NULL;
	double *sigmaSTGPU = NULL;
	cudaMalloc(&sigmaSGPU, smpheight * sizeof(double));
	cudaMalloc(&sigmaSTGPU, smpheight * sizeof(double));
	
	// get the sigmaST for the whole image
	double *sigmaSTForAll = NULL;
	cudaMalloc(&sigmaSTForAll, srcwidth * srcheight * sizeof(double));
	
	if(srcimageGPU == NULL || smpimageGPU == NULL) {
		printf("Fail to malloc new space in GPU\n");
	} else {
		printf("Sucess to malloc new space in GPU\n");
	}
	cudaMemcpy(srcimageGPU, srcimage, srcwidth * srcheight * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(smpimageGPU, smpimage, smpwidth * smpheight * sizeof(double), cudaMemcpyHostToDevice);
	
	// Execute the kernel
	double sigmaST = 1.0;
	double R = 1.0;
	double MaxR = -1;

	struct stopwatch_t* timer = NULL;
	long double t_kernel_0;
	stopwatch_init();
	timer = stopwatch_create();
	stopwatch_start(timer);
	printf("used kernel2.\n");

	kernel2(srcimageGPU, smpimageGPU, srcwidth, srcheight, smpwidth, smpheight, sigmaSTForAll);	
	// get all sigma on the GPU
	double *sigmaSTForAllCPU = new double[srcwidth * srcheight];
	cudaMemcpy(sigmaSTForAllCPU, sigmaSTForAll, srcwidth * srcheight * sizeof(double), cudaMemcpyDeviceToHost);

	for(int i = 0; i < srcheight - smpheight; ++ i) {
		for(int j = 0; j < srcwidth - smpwidth; ++ j) {
			double tempSigmaS = sumSigmaS[(i + smpheight - 1) * srcwidth + j + smpwidth - 1];
			if(i > 0) tempSigmaS -= sumSigmaS[(i - 1) * srcwidth + j + smpwidth - 1];
			if(j > 0) tempSigmaS -= sumSigmaS[(i + smpheight - 1) * srcwidth + j - 1];
			if(i > 0 && j > 0) tempSigmaS += sumSigmaS[(i - 1) * srcwidth + j - 1];
			
			sigmaST = sigmaSTForAllCPU[i * srcwidth + j];
			R = sigmaST /(sqrt(tempSigmaS) * sqrt(sigmaT));
			if(R > MaxR) {
				MaxR = R;
				x = i;
				y = j;
			}

		}
	}
	t_kernel_0 = stopwatch_stop(timer);
	printf("Running Time: %Lg\n", t_kernel_0);
	printf("MaxR = %lf\n", MaxR);	
	printf("x = %d, y = %d\n", x, y);

	return 0;
}



