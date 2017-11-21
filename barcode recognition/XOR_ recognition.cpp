#include "stdafx.h"
#include "XOR_ recognition.h"

struct result {
	double bi;
	int nu;

	bool operator<(result &m) {
		if (bi < m.bi)return true;
		else return false;
	}
}XOR_result[100];

char filename0[1000];

Mat nu[100];
Mat dit[10];
Mat sample;


void Threshold(Mat &src, Mat &sample, int m, int digit_xor)
{
	cvtColor(sample, sample, COLOR_BGR2GRAY);

	threshold(sample, sample, 150, 255, THRESH_BINARY | THRESH_OTSU);

	bitwise_not(sample, sample);




	XOR_result[m].bi = compare(src, sample, digit_xor);
	XOR_result[m].nu = m;
}

void deal(Mat &src, int& digit_xor)
{


	for (int i = 1; i <= 100; i++)
	{

		sprintf(filename0, "C:/Users/user/Documents/Visual Studio 2015/Projects/barcode recognition/barcode recognition/digits_train/new_digits/n%d.jpg", i);



		Mat sample = imread(filename0, 1);
		if (!sample.data) break;
		if ((src.rows) != 0 || (src.cols) != 0)
		{
			Threshold(src, sample, i, digit_xor);
		}


	}



	sort(XOR_result, XOR_result + 100);


	if (XOR_result[99].nu <11)
	{
		digit_xor = 0;
	}

	else if (10 <XOR_result[99].nu && XOR_result[99].nu <21)
	{
		digit_xor = 1;
	}

	else if (20 <XOR_result[99].nu && XOR_result[99].nu <31)
	{
		digit_xor = 2;
	}

	else if (30 <XOR_result[99].nu && XOR_result[99].nu <41)
	{
		digit_xor = 3;
	}

	else if (40 <XOR_result[99].nu && XOR_result[99].nu <51)
	{
		digit_xor = 4;
	}

	else if (50 <XOR_result[99].nu && XOR_result[99].nu <61)
	{
		digit_xor = 5;
	}

	else if (60 <XOR_result[99].nu && XOR_result[99].nu <71)
	{
		digit_xor = 6;
	}

	else if (70 <XOR_result[99].nu && XOR_result[99].nu <81)
	{
		digit_xor = 7;
	}

	else if (80 <XOR_result[99].nu && XOR_result[99].nu <91)
	{
		digit_xor = 8;
	}

	else if (90 <XOR_result[99].nu && XOR_result[99].nu <101)
	{
		digit_xor = 9;
	}

	/*cout << "最相似 digit_xor 為 " << digit_xor << endl;
	cout << "最相似為 " << XOR_result[99].nu << endl;
	cout << "識別度為 " << XOR_result[99].bi << endl;*/

}

double compare(Mat &src, Mat &sample, int digit_xor)
{
	double same = 0.0, difPoint = 0.0;
	Mat now, digits_tada;

	cv::resize(sample, now, cv::Size(12, 18));



	/*cvtColor(src, digits_tada, COLOR_BGR2GRAY);

	threshold(digits_tada, digits_tada, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

	bitwise_not(src, digits_tada);

	cv::resize(digits_tada, digits_tada, cv::Size(12, 18));



	int row = now.rows;
	int col = now.cols *  now.channels();
	for (int i = 0; i < 1; i++) {
		uchar * data1 = digits_tada.ptr<uchar>(i);
		uchar * data2 = now.ptr<uchar>(i);
		for (int j = 0; j < row * col; j++) {
			int  a = data1[j];
			int b = data2[j];
			if (a == b)same++;
			else difPoint++;
		}
	}
	return same / (same + difPoint);

}