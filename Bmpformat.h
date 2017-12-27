#ifndef BMP_H_
#define BMP_H_

#include <stdio.h> 
#include <string.h>

typedef unsigned char BYTE;
typedef unsigned short WORD;
typedef unsigned int DWORD;
typedef long LONG;

//Bitmap file header definition    
typedef struct  tagBITMAPFILEHEADER { 
	DWORD bfSize;//File size 
	WORD bfReserved1;//Reserved words  
	WORD bfReserved2;//Reserved words  
	DWORD bfOffBits;//The number of offset bytes from the header of the file to the actual bitmap data 
}BITMAPFILEHEADER;

//Bitmap information header definition
typedef struct tagBITMAPINFOHEADER {
	DWORD biSize;//Header size  
	DWORD biWidth;//Image width  
	DWORD biHeight;//Image height  
	WORD biPlanes;//The number of planes 
	WORD biBitCount;//Per pixel number  
	DWORD  biCompression; //Compression type 
	DWORD  biSizeImage; //Compressed image size bytes 
	DWORD  biXPelsPerMeter; //Horizontal resolution  
	DWORD  biYPelsPerMeter; //Vertical resolution 
	DWORD  biClrUsed; //The actual number of colors used by the bitmap  
	DWORD  biClrImportant; //The number of important colors in this bitmap  
}BITMAPINFOHEADER; 

typedef struct tagRGBQUAD {
	BYTE rgbBlue; //blue  
	BYTE rgbGreen; //green 
	BYTE rgbRed; //red 
	BYTE rgbReserved; //Reserved 
}RGBQUAD;//Palette  

#endif
