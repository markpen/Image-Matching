#include "BmpProcess.h"

BYTE* readBmp(char* fileName, int* width, int* height) {
	BITMAPFILEHEADER strHead;
	RGBQUAD strPla[256];//256 color Palette  
	BITMAPINFOHEADER strInfo;

	BYTE *imagedata = NULL;

	FILE *fpi = fopen(fileName, "rb");

	if (fpi != NULL) {
		//read file type
		WORD bfType;
		fread(&bfType, 1, sizeof(WORD), fpi);
		if (0x4d42 != bfType) //BM
		{
			printf("The file is not a bmp file!\n ");
			fclose(fpi);
			return NULL;
		}


		//read bmp file header and information header
		fread(&strHead, 1, sizeof(BITMAPFILEHEADER), fpi);
		

		fread(&strInfo, 1, sizeof(BITMAPINFOHEADER), fpi);	

		if (strInfo.biBitCount != 24) {
			printf("biBitCount is not 24!\n");
			fclose(fpi);
			return NULL;
		}

		printf("Image size: %d\n", (int)(strHead.bfSize - strHead.bfOffBits));

		long size = strHead.bfSize - strHead.bfOffBits;

		//read Palette
		for (unsigned int nCounti = 0; nCounti < strInfo.biClrUsed; nCounti++)
		{
			fread((char *)&(strPla[nCounti].rgbBlue), 1, sizeof(BYTE), fpi);
			fread((char *)&(strPla[nCounti].rgbGreen), 1, sizeof(BYTE), fpi);
			fread((char *)&(strPla[nCounti].rgbRed), 1, sizeof(BYTE), fpi);
			fread((char *)&(strPla[nCounti].rgbReserved), 1, sizeof(BYTE), fpi);
		}

		(*width) = strInfo.biWidth;
		(*height) = strInfo.biHeight;

		imagedata = new BYTE[size];
		fread(imagedata, sizeof(BYTE), size, fpi);

	}
	else {
		printf("file open error!\n");
		return NULL;
	}
	fclose(fpi);

	return imagedata;
}

bool writeBmp(char* fileName, BYTE* imagedata, int width, int height, long size) {
	FILE* fpw;
	// declare bmp structures 
	BITMAPFILEHEADER strHead;
	RGBQUAD strPla[256];//256 color Palette  
	BITMAPINFOHEADER strInfo;

	// andinitialize them to zero
	memset(&strHead, 0, sizeof(BITMAPFILEHEADER));
	memset(&strInfo, 0, sizeof(BITMAPINFOHEADER));

	// fill the fileheader with data
	WORD bfType = 0x4d42;
	//strHead.bfType = 0x4d42;       // 0x4d42 = 'BM'
	strHead.bfReserved1 = 0;
	strHead.bfReserved2 = 0;
	strHead.bfSize = sizeof(WORD) + sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + size;
	strHead.bfOffBits = 0x36;		// number of bytes to start of bitmap bits

									// fill the infoheader
	strInfo.biSize = sizeof(BITMAPINFOHEADER);
	strInfo.biWidth = width;
	strInfo.biHeight = height;
	strInfo.biPlanes = 1;			// we only have one bitplane
	strInfo.biBitCount = 24;		// RGB mode is 24 bits
	strInfo.biCompression = 0L;       //BI_RGB
	strInfo.biSizeImage = 0;		// can be 0 for 24 bit images
	strInfo.biXPelsPerMeter = 0x0ec4;     // paint and PSP use this values
	strInfo.biYPelsPerMeter = 0x0ec4;
	strInfo.biClrUsed = 0;			// we are in RGB mode and have no palette
	strInfo.biClrImportant = 0;    // all colors are important

	if ((fpw = fopen(fileName, "wb")) == NULL)
	{
		printf("create the bmp file error!\n");
		return false;
	}
	fwrite(&bfType, 1, sizeof(WORD), fpw);
	fwrite(&strHead, 1, sizeof(tagBITMAPFILEHEADER), fpw);
	fwrite(&strInfo, 1, sizeof(tagBITMAPINFOHEADER), fpw);

	for (unsigned int nCounti = 0; nCounti<strInfo.biClrUsed; nCounti++)
	{
		fwrite(&strPla[nCounti].rgbBlue, 1, sizeof(BYTE), fpw);
		fwrite(&strPla[nCounti].rgbGreen, 1, sizeof(BYTE), fpw);
		fwrite(&strPla[nCounti].rgbRed, 1, sizeof(BYTE), fpw);
		fwrite(&strPla[nCounti].rgbReserved, 1, sizeof(BYTE), fpw);
	}

	fwrite(imagedata, size, sizeof(BYTE), fpw);
	fclose(fpw);

	return true;
}

BYTE* ConvertBMPToRGBBuffer(BYTE* Buffer, int* width, int* height) {
	// first make sure the parameters are valid
	if ((NULL == Buffer) || (width == 0) || (height == 0))
		return NULL;

	// find the number of padding bytes

	int padding = 0;
	int scanlinebytes = (*width) * 3;
	while ((scanlinebytes + padding) % 4 != 0)     // DWORD = 4 bytes
		padding++;
	// get the padded scanline width
	int psw = scanlinebytes + padding;

	// create new buffer
	BYTE* newbuf = new BYTE[(*width) * (*height) * 3];

	// now we loop trough all bytes of the original buffer, 
	// swap the R and B bytes and the scanlines
	long bufpos = 0;
	long newpos = 0;

	for (int y = 0; y < (*height); y++)
		for (int x = 0; x < 3 * (*width); x += 3) {
			newpos = y * 3 * (*width) + x;
			bufpos = ((*height) - y - 1) * psw + x;

			newbuf[newpos] = Buffer[bufpos + 2];
			newbuf[newpos + 1] = Buffer[bufpos + 1];
			newbuf[newpos + 2] = Buffer[bufpos];
		}

	return newbuf;
}

BYTE* ConvertRGBToBMPBuffer(BYTE* Buffer, int width, int height, long* newsize) {

	// first make sure the parameters are valid
	if ((NULL == Buffer) || (width == 0) || (height == 0))
		return NULL;

	// now we have to find with how many bytes
	// we have to pad for the next DWORD boundary	

	int padding = 0;
	int scanlinebytes = width * 3;
	while ((scanlinebytes + padding) % 4 != 0)     // DWORD = 4 bytes
		padding++;
	// get the padded scanline width
	int psw = scanlinebytes + padding;

	// we can already store the size of the new padded buffer
	*newsize = height * psw;

	// and create new buffer
	BYTE* newbuf = new BYTE[*newsize];

	// fill the buffer with zero bytes then we dont have to add
	// extra padding zero bytes later on
	memset(newbuf, 0, *newsize);

	// now we loop trough all bytes of the original buffer, 
	// swap the R and B bytes and the scanlines
	long bufpos = 0;
	long newpos = 0;
	for (int y = 0; y < height; y++)
		for (int x = 0; x < 3 * width; x += 3) {
			bufpos = y * 3 * width + x;     // position in original buffer
			newpos = (height - y - 1) * psw + x;           // position in padded buffer

			newbuf[newpos] = Buffer[bufpos + 2];       // swap r and b
			newbuf[newpos + 1] = Buffer[bufpos + 1]; // g stays
			newbuf[newpos + 2] = Buffer[bufpos];     // swap b and r
		}

	return newbuf;
}
void showMatchposition(BYTE* rgbBuffer, int width1, int height1, int width2, int height2, int x, int y) {
	if (rgbBuffer == NULL) return;
	for (int i = 3 * (width1*y + x); i <= 3 * (width1*y + x + width2); i += 3) {//up
		rgbBuffer[i] = (BYTE)255;
		rgbBuffer[i + 1] = (BYTE)0;
		rgbBuffer[i + 2] = (BYTE)0;
	}
	for (int i = 3 * (width1*(y + 1) + x); i < 3 * (width1*(y + height2) + x); i += 3 * width1) {//left
		rgbBuffer[i] = (BYTE)255;
		rgbBuffer[i + 1] = (BYTE)0;
		rgbBuffer[i + 2] = (BYTE)0;
	}
	for (int i = 3 * (width1*(y + 1) + x + width2); i < 3 * (width1*(y + height2) + x + width2); i += 3 * width1) {//right
		rgbBuffer[i] = (BYTE)255;
		rgbBuffer[i + 1] = (BYTE)0;
		rgbBuffer[i + 2] = (BYTE)0;
	}
	for (int i = 3 * (width1*(y + height2) + x); i <= 3 * (width1*(y + height2) + x + width2); i += 3) {//down
		rgbBuffer[i] = (BYTE)255;
		rgbBuffer[i + 1] = (BYTE)0;
		rgbBuffer[i + 2] = (BYTE)0;
	}
	long size = 3 * width1*height1;
	BYTE *match = ConvertRGBToBMPBuffer(rgbBuffer, width1, height1, &size);
	writeBmp("x.bmp", match, width1, height1, size);
	delete[] match;
}

BYTE* RGBtoGray(BYTE *rgbBuffer, int* width, int* height) {
	BYTE *grayimage = NULL;
	if (rgbBuffer == NULL) return NULL;

	//BYTE *rgbBuffer = ConvertBMPToRGBBuffer(imagedata, width, height);

	/*long size = 3 * (*width)*(*height);
	BYTE *test = ConvertRGBToBMPBuffer(rgbBuffer, (*width), (*height), &size);
	writeBmp("x.bmp", test, (*width), (*height), size);*/

	if (rgbBuffer != NULL) {
		grayimage = new BYTE[(*width) * (*height)];
			
		//#pragma omp parallel for
		for (int i = 0; i < 3 * (*width) * (*height); i += 3) {
			int r = (int)rgbBuffer[i];
			int g = (int)rgbBuffer[i + 1];
			int b = (int)rgbBuffer[i + 2];
			grayimage[i/3] = (r * 299 + g * 587 + b * 114 + 500) / 1000;
		}
	}
	else
	{
		printf("Cannot convert BMP to RGB\n");
		//cout << "Can't convert BMP to RGB!" << endl;
		return NULL;
	}

	//delete[] rgbBuffer;	

	return grayimage;
}

