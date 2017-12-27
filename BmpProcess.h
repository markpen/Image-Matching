#ifndef BmpProcess_H_
#define BProcess_H_

#include "Bmpformat.h"
#include <iostream> 

BYTE* readBmp(char* fileName, int* width, int* height);
bool writeBmp(char* fileName, BYTE* imagedata, int width, int height, long size);
BYTE* ConvertBMPToRGBBuffer(BYTE* Buffer, int* width, int* height);
void showMatchposition(BYTE* rgbBuffer, int width1, int height1, int width2, int height2, int x, int y);

#endif