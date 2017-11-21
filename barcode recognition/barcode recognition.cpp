#include "stdafx.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp" 
#include "opencv2/core/core.hpp"
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <stack>  
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <algorithm>
#include <time.h>
#include <cstring> 
#include <opencv2/ml/ml.hpp>
#include <sstream>
#include <algorithm>
#include "XOR_ recognition.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

namespace cv
{
	using std::vector;
}

#define SPACE 0
#define bar 255
const int ERROR_CODE = 9999999;


typedef Mat_<uchar> MatU;
map<string, int> table[3];
enum position { LEFT, RIGHT };
int front;
int mode[6];
const double eps = 1e-5;
int z;

char filename[1000];
int success = 0, test = 0, one_success = 0, two_success = 0, three_success = 0, four_success = 0, digits_success = 0, digits1_success = 0,
digits2_success = 0, onltsuc1 = 0, onltsuc2 = 0, onltsuc = 0, all_sean = 0, ground_truth = 0, bar_or_digits = 0, ground_truth_digits = 0;
int  check_num[13] = { 0 };

int one_read = 0, two_read = 0, three_read = 0, four_read = 0, five_read = 0, six_reading = 0;


int digit_xor = 0;
int six_read = 0;
int  svm_digit = 0;

int reseg = 0;

char Name[20];

vector<string> digitresult[1];
template <class T>
void convertFromString(T &, const std::string &);



const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 12;
const int RESIZED_IMAGE_HEIGHT = 18;

void read_barcode4(MatU&, MatU&, MatU&, MatU&, MatU&, MatU&, int& reseg, int& img_num);
void read_barcode3(MatU&, MatU&, MatU&, MatU&, MatU&, MatU&, int& reseg, int& img_num);
void read_barcode2(MatU&, MatU&, MatU&, MatU&, MatU&, MatU&, int& reseg, int& img_num);
void read_barcode(MatU&, MatU&, MatU&, MatU&, MatU&, MatU&, int& reseg, int& img_num);
void seg_barcod(Mat&);

void xor_digitsort(Mat &, Mat &, Mat&, Mat&, int&);
void xor_again_digitsort(Mat &, Mat &, Mat&, Mat&, int&);
void xor_three_digitsort(Mat &, Mat &, Mat&, Mat&, int&);
///////////////////////////////////////////////////////////////////////////////////////////////////
bool myfunction(int i, int j) { return (i<j); }

struct myclass_hiss {
	bool operator() (int i, int j) { return (i<j); }
} myobject;



template <class T>
void convertFromString(T &value, const std::string &s)  //
{
	std::stringstream ss(s);
	ss >> value;

}



class ContourWithData {
public:
	// member variables ///////////////////////////////////////////////////////////////////////////
	std::vector<cv::Point> ptContour;           // contour
	cv::Rect boundingRect;                      // bounding rect for contour
	float fltArea;                              // area of contour

												///////////////////////////////////////////////////////////////////////////////////////////////
	bool checkIfContourIsValid() {                              // obviously in a production grade program
		if (fltArea < MIN_CONTOUR_AREA) return false;           // we would have a much more robust function for 
		return true;                                            // identifying if a contour is valid !!
	}

	///////////////////////////////////////////////////////////////////////////////////////////////
	static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) {      // this function allows us to sort
		return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);                                                   // the contours from left to right
	}

};

int get_front() { /////////////國家代碼
	string tmp = "";
	int i;
	for (i = 0; i < 6; i++) {
		if (mode[i] == 0) tmp = tmp + "0";
		else tmp = tmp + "1";
	}
	if (tmp == "000000") return 0;
	else if (tmp == "001011") return 1;
	else if (tmp == "001101") return 2;
	else if (tmp == "001110") return 3;
	else if (tmp == "010011") return 4;
	else if (tmp == "011001") return 5;
	else if (tmp == "011100") return 6;
	else if (tmp == "010101") return 7;
	else if (tmp == "010110") return 8;
	else if (tmp == "011010") return 9;
	else return -99;
}

//*int read_digit(...)

int read_digit(const MatU& img, Point& cur, int position) {
	int pattern[4] = { 0, 0, 0, 0 };
	int i;
	for (i = 0; i < 4; i++) {   ///每個字元由2條Space及2條Bar間隔組成
		int cur_val = img(cur);
		while (img(cur) == cur_val) {
			++pattern[i];

			if (cur.x < img.cols - 1)
				++cur.x;
			else
				return ERROR_CODE;
		}
	}
	double sum = pattern[0] + pattern[1] + pattern[2] + pattern[3];
	double tmp1 = (pattern[0] + pattern[1])*1.0;
	double tmp2 = (pattern[1] + pattern[2])*1.0;
	int at1, at2;
	if (tmp1 / sum < 2.5 / 7)  at1 = 2;
	else if (tmp1 / sum < 3.5 / 7) at1 = 3;
	else if (tmp1 / sum < 4.5 / 7) at1 = 4;
	else at1 = 5;

	if (tmp2 / sum < 2.5 / 7)  at2 = 2;
	else if (tmp2 / sum < 3.5 / 7) at2 = 3;
	else if (tmp2 / sum < 4.5 / 7) at2 = 4;
	else at2 = 5;

	int digit = -999;

	if (position == LEFT) {  ///////////////////////////////////////////////////////////////左資料區
		if (at1 == 2) {
			if (at2 == 2) {
				mode[z++] = 0;
				digit = 6;
			}
			else if (at2 == 3) {
				mode[z++] = 1;
				digit = 0;
			}
			else if (at2 == 4) {
				mode[z++] = 0;
				digit = 4;
			}
			else if (at2 == 5) {
				mode[z++] = 1;
				digit = 3;
			}
		}
		else if (at1 == 3) {
			if (at2 == 2) {
				mode[z++] = 1;
				digit = 9;
			}
			else if (at2 == 3) {
				mode[z++] = 0;
				if (pattern[2] + 1 < pattern[3]) digit = 8;
				else digit = 2;
			}
			else if (at2 == 4) {
				mode[z++] = 1;
				if (pattern[1] + 1 < pattern[2]) digit = 7;
				else digit = 1;
			}
			else if (at2 == 5) {
				mode[z++] = 0;
				digit = 5;
			}
		}
		else if (at1 == 4) {
			if (at2 == 2) {
				mode[z++] = 0;
				digit = 9;
			}
			else if (at2 == 3) {
				mode[z++] = 1;
				if (pattern[1] + 1 < pattern[0]) digit = 8;
				else digit = 2;
			}
			else if (at2 == 4) {
				mode[z++] = 0;
				if (pattern[0] + 1 < pattern[1]) digit = 7;
				else digit = 1;
			}
			else if (at2 == 5) {
				mode[z++] = 1;
				digit = 5;
			}
		}
		else if (at1 == 5) {
			if (at2 == 2) {
				mode[z++] = 1;
				digit = 6;
			}
			else if (at2 == 3) {
				mode[z++] = 0;
				digit = 0;
			}
			else if (at2 == 4) {
				mode[z++] = 1;
				digit = 4;
			}
			else if (at2 == 5) {
				mode[z++] = 0;
				digit = 3;
			}
		}

	}
	else {           ///////////////////////////////////////////////////////////////右資料區
		if (at1 == 2) {
			if (at2 == 2) digit = 6;
			else if (at2 == 4) digit = 4;
		}
		else if (at1 == 3) {
			if (at2 == 3) {
				if (pattern[2] + 1 < pattern[3]) digit = 8;
				else digit = 2;
			}
			else if (at2 == 5) digit = 5;
		}
		else if (at1 == 4) {
			if (at2 == 2) digit = 9;
			else if (at2 == 4) {
				if (pattern[0] + 1 < pattern[1]) digit = 7;
				else digit = 1;
			}
		}
		else if (at1 == 5) {
			if (at2 == 3) digit = 0;
			else if (at2 == 5) digit = 3;
		}

	}
	return digit;
}

void skip_quiet_zone(const MatU& img, Point& cur) { ////略過空白區
	while (img(cur) == SPACE)
		if (cur.x < img.cols - 1)
			++cur.x;
		else
			break;
}

void read_lguard(const MatU& img, Point& cur) { ///讀左護線 計算條寬
	int pattern[3] = { bar, SPACE, bar };
	for (int i = 0; i < 3; i++)
		while (img(cur) == pattern[i])
			if (cur.x < img.cols - 1)
				++cur.x;
			else
				break;
}

void skip_mguard(const MatU& img, Point& cur) { ////略過中線
	int pattern[5] = { SPACE, bar, SPACE, bar, SPACE };
	for (int i = 0; i < 5; i++)
		while (img(cur) == pattern[i])
			if (cur.x < img.cols - 1)
				++cur.x;
			else
				break;
}

void Sharpening(cv::Mat getpixel, int Mask[9][2], cv::Mat& Matbox)
{

	int Box[9] = { 1, 1, 1,
		1, 1, 1,
		1, 1, 1, };


	for (int h = 1; h < getpixel.rows - 2; h++)
	{
		for (int w = 1; w < getpixel.cols - 2; w++)
		{
			int sum = 0;

			for (int m = 0; m < 9; m++)
			{
				sum = sum + (getpixel.at<unsigned char>(h + Mask[m][0], w + Mask[m][1]) / 9);

			}
			if (sum < 0) sum = 0;
			if (sum > 255) sum = 255;
			Matbox.at<unsigned char>(h, w) = sum;
		}
	}
}

void sharpenImage1(const cv::Mat &image, cv::Mat &result)
{

	cv::Mat kernel(3, 3, CV_32F, cv::Scalar(0));
	kernel.at<float>(1, 1) = 5.0;
	kernel.at<float>(0, 1) = -1.0;
	kernel.at<float>(1, 0) = -1.0;
	kernel.at<float>(1, 2) = -1.0;
	kernel.at<float>(2, 1) = -1.0;

	result.create(image.size(), image.type());


	cv::filter2D(image, result, image.depth(), kernel);
}


class myclass {
public:
	myclass(int a, int b, int c, int d) :first(a), second(b), third(c), fourth(d) {}
	int first;
	int second, third, fourth;
	bool operator < (const myclass &m)const {
		return first < m.first;
	}
};


class Location {
public:
	Location(int a, int b, int c, int d, int e) :first(a), second(b), third(c), fourth(d), fifth(e) {}
	int first;
	int second, third, fourth, fifth;
	bool operator < (const Location &m)const {
		return first > m.first;
	}
};


class projection {
public:
	projection(int a, int b, int c, int d, int e, int f, int g, int h, int i, int j, int k, int l, int m, int n, int o, int p, int q,
		int r, int s, int t, int u, int v, int w, int x, int y, int z) :f1(a), f2(b), f3(c), f4(d), f5(e), f6(f), f7(g), f8(h),
		f9(i), f10(j), f11(k), f12(l), f13(m), f14(n), f15(o), f16(p), f17(q), f18(r), f19(s), f20(t), f21(u), f22(v), f23(w), f24(x),
		f25(y), f26(z) {}


	int f1;
	int f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26;
	bool operator < (const projection &m)const {
		return f1 > m.f1;
	}
};

void check(int check_num[], int img_num)
{
	int answer_arary[100][13] = {
		0,2,8,2,9,2,5,0,3,7,1,9,8,
		2,0,0,5,1,6,9,2,3,3,0,0,7,
		4,0,0,0,6,8,0,7,0,3,2,6,6,
		9,0,0,2,7,4,0,2,1,8,2,3,4,
		9,0,0,2,7,5,9,2,5,0,0,6,5,
		9,0,0,2,7,5,9,7,1,8,8,0,0,
		9,0,0,2,7,5,9,8,1,2,6,1,4,
		9,0,0,2,7,5,9,8,4,0,3,5,8,
		9,0,0,2,8,2,7,2,4,9,0,1,4,
		9,0,0,5,4,3,7,7,5,0,0,3,1,

		9,0,0,6,5,0,5,1,2,3,0,4,7,
		9,0,0,6,9,0,0,0,1,1,5,2,9,
		9,0,0,6,9,3,6,0,1,9,5,6,8,
		4,0,0,0,8,2,3,0,6,1,1,0,9,
		9,0,0,7,0,8,4,0,2,3,2,5,4,
		9,0,0,8,4,2,0,5,0,1,2,3,8,
		9,0,1,1,5,0,0,0,2,5,0,4,3,
		9,0,8,8,8,8,2,4,4,8,2,0,3,
		9,3,1,0,0,0,6,0,0,6,6,4,9,
		9,3,2,2,2,1,4,0,0,6,2,2,9,

		4,0,0,0,8,2,3,0,6,1,1,0,9,
		4,0,0,0,8,6,9,0,4,2,0,5,6,
		4,0,0,0,8,6,9,0,6,0,0,9,8,
		4,0,0,0,8,6,9,0,6,0,0,9,8,
		4,0,0,0,8,7,0,0,3,6,9,8,3,
		4,0,0,0,8,7,0,9,8,4,6,1,1,
		4,0,0,0,9,5,3,3,9,8,9,1,5,
		4,0,0,0,9,9,9,9,1,4,1,2,4,
		2,0,0,7,0,0,1,4,3,7,5,5,0,
		4,0,0,1,0,4,4,1,7,0,1,8,2,

		4,0,0,1,1,2,0,2,6,0,9,0,5,
		4,0,0,1,1,3,3,2,0,5,5,1,1,
		4,0,0,1,1,3,3,2,4,5,8,1,4,
		4,0,0,1,1,3,3,2,9,5,9,1,8,
		4,0,0,1,1,3,3,3,2,3,0,1,7,
		4,0,0,1,1,3,3,3,2,8,6,6,1,
		4,0,0,1,1,3,3,5,8,4,5,1,7,
		4,0,0,1,1,3,3,8,3,8,6,2,7,
		4,0,0,1,1,3,3,8,4,8,6,1,9,
		2,0,0,7,0,0,1,5,2,3,2,3,9,

		4,0,0,1,1,3,3,8,7,1,8,1,5,
		4,0,0,1,1,3,3,9,2,9,3,1,8,
		4,0,0,1,1,3,3,9,2,9,5,1,6,
		4,0,0,1,1,3,3,9,6,2,8,1,0,
		4,0,0,1,1,6,7,7,9,9,0,5,5,
		4,0,0,1,1,6,7,8,0,4,1,4,8,
		4,0,0,1,4,1,9,0,0,9,6,1,1,
		4,0,0,1,4,3,2,0,6,9,5,3,1,
		4,0,0,1,4,3,2,0,7,8,5,3,3,
		4,0,0,1,4,3,8,0,1,2,1,3,5,

		2,0,0,7,0,0,1,5,7,2,3,1,2,
		4,0,0,1,4,7,8,0,0,2,7,4,5,
		4,0,0,1,4,9,7,2,8,7,2,0,8,
		4,0,0,1,4,9,7,2,8,7,2,0,8,
		4,0,0,1,4,9,7,3,2,1,3,0,8,
		4,0,0,1,4,9,9,0,1,2,3,5,8,
		4,0,0,1,4,9,9,1,4,0,6,4,8,
		4,0,0,1,4,9,9,1,4,3,6,9,4,
		4,0,0,1,4,9,9,1,6,2,9,5,4,
		4,0,0,1,5,1,3,0,0,0,6,1,3,

		4,0,0,1,5,1,8,0,0,6,4,5,0,
		2,0,0,7,0,0,1,5,7,2,3,1,2,
		4,0,0,1,5,1,8,0,0,7,8,8,4,
		4,0,0,1,6,0,8,0,2,3,5,1,0,
		4,0,0,1,6,0,8,2,0,2,3,5,9,
		4,0,0,1,6,0,8,2,0,3,9,8,1,
		4,0,0,1,6,0,8,2,0,4,0,2,5,
		4,0,0,1,6,0,8,2,0,4,0,4,9,
		4,0,0,1,6,0,8,2,0,4,1,3,1,
		4,0,0,1,6,8,3,0,0,1,1,3,5,

		4,0,0,1,7,0,2,0,2,5,3,0,4,
		4,0,0,1,7,2,3,1,4,5,0,0,5,
		2,0,0,7,0,0,1,8,8,6,4,1,9,
		4,0,0,1,7,4,3,0,7,9,2,5,0,
		4,0,0,1,8,4,5,5,2,1,1,0,7,
		4,0,0,1,8,4,5,5,2,4,3,6,8,
		4,0,0,1,8,4,5,5,2,5,1,4,3,
		4,0,0,1,8,4,5,5,2,5,2,0,4,
		4,0,0,1,8,4,5,5,2,5,2,6,6,
		4,0,0,1,8,4,5,5,2,7,0,0,0,

		4,0,0,1,8,8,3,2,9,4,6,6,7,
		4,0,0,1,9,3,6,0,1,1,4,0,1,
		4,0,0,1,9,5,6,2,1,1,9,8,0,
		2,0,0,7,0,0,2,0,0,6,2,1,2,
		4,0,0,1,9,5,6,5,4,5,0,1,6,
		4,0,0,2,0,2,5,0,5,0,3,8,7,
		4,0,0,2,0,3,9,2,1,6,3,0,4,
		4,0,0,2,0,6,4,4,0,8,7,5,0,
		4,0,0,2,2,3,9,4,2,2,6,0,4,
		4,0,0,2,2,3,9,5,9,5,5,0,6,

		4,0,0,2,3,5,9,1,7,6,3,0,2,
		4,0,0,2,4,4,1,0,3,9,9,6,6,
		4,0,0,2,4,4,8,0,0,6,4,7,3,
		4,0,0,2,4,4,8,0,1,6,2,2,9,
		2,0,0,7,0,0,2,1,2,4,8,6,2,
		4,0,0,2,4,4,8,0,1,8,3,6,0,
		4,0,0,2,4,4,8,0,1,8,5,5,1,
		4,0,0,2,4,4,8,0,2,4,2,1,7,
		4,0,0,2,4,4,8,0,2,7,1,4,0,
		4,0,0,2,4,4,8,0,2,7,5,0,8,

	};

	switch (bar_or_digits)
	{

	case 1:

		if (answer_arary[img_num - 1][0] == check_num[0] && answer_arary[img_num - 1][1] == check_num[1] && answer_arary[img_num - 1][2] == check_num[2] && answer_arary[img_num - 1][3] == check_num[3] && answer_arary[img_num - 1][4] == check_num[4]
			&& answer_arary[img_num - 1][5] == check_num[5] && answer_arary[img_num - 1][6] == check_num[6] && answer_arary[img_num - 1][7] == check_num[7] && answer_arary[img_num - 1][8] == check_num[8] && answer_arary[img_num - 1][9] == check_num[9] &&
			answer_arary[img_num - 1][10] == check_num[10] && answer_arary[img_num - 1][11] == check_num[11] && answer_arary[img_num - 1][12] == check_num[12])
		{
			cout << "條紋解碼結果與答案相符" << endl;
		}
		else
		{
			cout << "條紋解碼結果與答案不相符" << endl;
			ground_truth = 1;
		}

		break;



	case 2:


		if (answer_arary[img_num - 1][0] == check_num[0] && answer_arary[img_num - 1][1] == check_num[1] && answer_arary[img_num - 1][2] == check_num[2] && answer_arary[img_num - 1][3] == check_num[3] && answer_arary[img_num - 1][4] == check_num[4]
			&& answer_arary[img_num - 1][5] == check_num[5] && answer_arary[img_num - 1][6] == check_num[6] && answer_arary[img_num - 1][7] == check_num[7] && answer_arary[img_num - 1][8] == check_num[8] && answer_arary[img_num - 1][9] == check_num[9] &&
			answer_arary[img_num - 1][10] == check_num[10] && answer_arary[img_num - 1][11] == check_num[11] && answer_arary[img_num - 1][12] == check_num[12])
		{
			cout << "數字辨識結果與答案相符" << endl;
		}
		else
		{
			cout << "數字辨識結果與答案不相符" << endl;
			ground_truth_digits = 1;
		}


		break;

	}




}



///////////////////////////////////////////////////////////////////////// 第六階段  
void six_read_barcode4(MatU& img, int& reseg, int& img_num)
{
	z = 0;


	MatU flip_img;



	flip(img, flip_img, 1);///水平翻轉

	int size = flip_img.rows;

	Point cur(0, (size / 2));  ///掃描位置



	skip_quiet_zone(flip_img, cur); //略過空白區
	read_lguard(flip_img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(flip_img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(flip_img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(flip_img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{

			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			four_success += 1;
			success += 1;
			onltsuc1 = 1;
			six_reading = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}


		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置



		skip_quiet_zone(flip_img, cur); //略過空白區
		read_lguard(flip_img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(flip_img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(flip_img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(flip_img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{

				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}


				cout << "檢查碼為正確" << endl;
				four_success += 1;
				success += 1;
				onltsuc1 = 1;
				six_reading = 1;

				bar_or_digits = 1;

				reseg = 1;

				check(check_num, img_num);
			}

			cout << endl;
		}
		else    //////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置



			skip_quiet_zone(flip_img, cur); //略過空白區
			read_lguard(flip_img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(flip_img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(flip_img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(flip_img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}

					cout << "檢查碼為正確" << endl;
					four_success += 1;
					success += 1;
					onltsuc1 = 1;
					six_reading = 1;

					reseg = 1;

					bar_or_digits = 1;

					check(check_num, img_num);
				}
				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "解碼錯誤" << endl;

			}

		}

	}

	//imshow("flip_image", flip_img);

}



/////////////////////////////////////////////////////////////////// 第六階段      第三次掃描

void six_read_barcode3(MatU& img, int& reseg, int& img_num)
{
	z = 0;




	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, (size / 4));  ///掃描位置
							   /*bitwise_not(img, img);
							   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							   //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{

			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			three_success += 1;
			success += 1;
			onltsuc1 = 1;
			six_reading = 1;

			reseg = 1;

			bar_or_digits = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第四次掃描" << endl;

			six_read_barcode4(img, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第四次掃描" << endl;

		six_read_barcode4(img, reseg, img_num);

	}



}


///////////////////////////////////////////////////////////////////////// 第六階段      第二次掃描
void  six_read_barcode2(MatU& img, int& reseg, int& img_num)
{
	z = 0;



	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, ((size / 4) * 3));  ///掃描位置
									 /*bitwise_not(img, img);
									 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{

			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			two_success += 1;
			success += 1;
			onltsuc1 = 1;
			six_reading = 1;

			reseg = 1;

			bar_or_digits = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第三次掃描" << endl;

			six_read_barcode3(img, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第三次掃描" << endl;

		six_read_barcode3(img, reseg, img_num);
	}

}


//////////////////////////////////////////////////////////////////////////第六階段     第一次掃描
void  six_read_barcode(MatU& img, int& reseg, int& img_num)
{
	z = 0;


	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, size / 2);  ///掃描位置
							 /*bitwise_not(img, img);
							 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			one_success += 1;
			success += 1;
			onltsuc1 = 1;
			six_reading = 1;

			reseg = 1;

			bar_or_digits = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第二次掃描" << endl;
			six_read_barcode2(img, reseg, img_num);
		}
		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置
										 /*bitwise_not(img, img);
										 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

										 //if (img(cur) != SPACE) return;


		skip_quiet_zone(img, cur); //略過空白區
		read_lguard(img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{
				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}

				reseg = 1;

				cout << "檢查碼為正確" << endl;
				two_success += 1;
				success += 1;
				onltsuc1 = 1;
				six_reading = 1;



				bar_or_digits = 1;

				check(check_num, img_num);
			}
			else
			{
				cout << endl;
				cout << "第三次掃描" << endl;

				six_read_barcode3(img, reseg, img_num);

			}
			cout << endl;
		}
		else  //////////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置
									   /*bitwise_not(img, img);
									   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									   //if (img(cur) != SPACE) return;


			skip_quiet_zone(img, cur); //略過空白區
			read_lguard(img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}


					cout << "檢查碼為正確" << endl;
					three_success += 1;
					success += 1;
					onltsuc1 = 1;
					six_reading = 1;

					reseg = 1;

					bar_or_digits = 1;

					check(check_num, img_num);
				}
				else
				{
					cout << endl;
					cout << "第四次掃描" << endl;

					six_read_barcode4(img, reseg, img_num);

				}
				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "第四次掃描" << endl;

				six_read_barcode4(img, reseg, img_num);

			}
		}
	}
	//waitKey();
}


///////////////////////////////////////////////////////////////////////////





///////////////////////////////////////////////////////////////////////// 第五階段  
void five_read_barcode4(MatU& img, MatU& barcode6, int& reseg, int img_num)
{
	z = 0;
	six_read = 0;

	MatU flip_img;



	flip(img, flip_img, 1);///水平翻轉

	int size = flip_img.rows;

	Point cur(0, (size / 2));  ///掃描位置



	skip_quiet_zone(flip_img, cur); //略過空白區
	read_lguard(flip_img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(flip_img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(flip_img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(flip_img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{

			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			four_success += 1;
			success += 1;
			onltsuc1 = 1;
			five_read = 1;
			six_read = 1;

			reseg = 1;

			bar_or_digits = 1;

			check(check_num, img_num);
		}


		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置



		skip_quiet_zone(flip_img, cur); //略過空白區
		read_lguard(flip_img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(flip_img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(flip_img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(flip_img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{

				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}


				cout << "檢查碼為正確" << endl;
				four_success += 1;
				success += 1;
				onltsuc1 = 1;
				five_read = 1;
				six_read = 1;

				reseg = 1;

				bar_or_digits = 1;

				check(check_num, img_num);
			}

			cout << endl;
		}
		else    //////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置



			skip_quiet_zone(flip_img, cur); //略過空白區
			read_lguard(flip_img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(flip_img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(flip_img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(flip_img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}

					cout << "檢查碼為正確" << endl;
					four_success += 1;
					success += 1;
					onltsuc1 = 1;
					five_read = 1;
					six_read = 1;

					bar_or_digits = 1;

					reseg = 1;

					check(check_num, img_num);
				}
				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "解碼錯誤" << endl;

			}

		}

	}


	//imshow("flip_image", flip_img);

	if (six_read == 0)
	{
		six_read_barcode(barcode6, reseg, img_num);
	}

}



/////////////////////////////////////////////////////////////////// 第五階段      第三次掃描

void five_read_barcode3(MatU& img, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;




	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, (size / 4));  ///掃描位置
							   /*bitwise_not(img, img);
							   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							   //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{

			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			three_success += 1;
			success += 1;
			onltsuc1 = 1;
			five_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第四次掃描" << endl;

			five_read_barcode4(img, barcode6, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第四次掃描" << endl;

		five_read_barcode4(img, barcode6, reseg, img_num);

	}



}


///////////////////////////////////////////////////////////////////////// 第五階段      第二次掃描
void  five_read_barcode2(MatU& img, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;



	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, ((size / 4) * 3));  ///掃描位置
									 /*bitwise_not(img, img);
									 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{

			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			two_success += 1;
			success += 1;
			onltsuc1 = 1;
			five_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第三次掃描" << endl;

			five_read_barcode3(img, barcode6, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第三次掃描" << endl;

		five_read_barcode3(img, barcode6, reseg, img_num);
	}

}


//////////////////////////////////////////////////////////////////////////第五階段     第一次掃描
void  five_read_barcode(MatU& img, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;


	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, size / 2);  ///掃描位置
							 /*bitwise_not(img, img);
							 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			one_success += 1;
			success += 1;
			onltsuc1 = 1;
			five_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第二次掃描" << endl;
			five_read_barcode2(img, barcode6, reseg, img_num);
		}
		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置
										 /*bitwise_not(img, img);
										 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

										 //if (img(cur) != SPACE) return;


		skip_quiet_zone(img, cur); //略過空白區
		read_lguard(img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{
				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}


				cout << "檢查碼為正確" << endl;
				two_success += 1;
				success += 1;
				onltsuc1 = 1;
				five_read = 1;

				bar_or_digits = 1;

				reseg = 1;

				check(check_num, img_num);
			}
			else
			{
				cout << endl;
				cout << "第三次掃描" << endl;

				five_read_barcode3(img, barcode6, reseg, img_num);

			}
			cout << endl;
		}
		else  //////////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置
									   /*bitwise_not(img, img);
									   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									   //if (img(cur) != SPACE) return;


			skip_quiet_zone(img, cur); //略過空白區
			read_lguard(img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}


					cout << "檢查碼為正確" << endl;
					three_success += 1;
					success += 1;
					onltsuc1 = 1;
					five_read = 1;

					bar_or_digits = 1;

					reseg = 1;

					check(check_num, img_num);
				}
				else
				{
					cout << endl;
					cout << "第四次掃描" << endl;

					five_read_barcode4(img, barcode6, reseg, img_num);

				}
				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "第四次掃描" << endl;

				five_read_barcode4(img, barcode6, reseg, img_num);

			}
		}
	}
	//waitKey();
}


///////////////////////////////////////////////////////////////////////////






///////////////////////////////////////////////////////////////////////// 第四階段  
void four_read_barcode4(MatU& img, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;


	MatU flip_img;

	MatU barcode_five = barcode5;

	flip(img, flip_img, 1);///水平翻轉

	int size = flip_img.rows;

	Point cur(0, (size / 2));  ///掃描位置



	skip_quiet_zone(flip_img, cur); //略過空白區
	read_lguard(flip_img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(flip_img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(flip_img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(flip_img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			four_success += 1;
			success += 1;
			onltsuc1 = 1;
			four_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);


		}
		else
		{
			cout << endl;
			cout << "第4階段 解碼錯誤" << endl;
			cout << "換銳化 區域OTSU " << endl;
			five_read_barcode(barcode_five, barcode6, reseg, img_num);

		}



		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置



		skip_quiet_zone(flip_img, cur); //略過空白區
		read_lguard(flip_img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(flip_img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(flip_img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(flip_img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{
				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}

				cout << "檢查碼為正確" << endl;
				four_success += 1;
				success += 1;
				onltsuc1 = 1;
				four_read = 1;

				bar_or_digits = 1;

				reseg = 1;

				check(check_num, img_num);

			}
			else
			{
				cout << endl;
				cout << "第4階段 解碼錯誤" << endl;
				cout << "換銳化 區域OTSU " << endl;
				five_read_barcode(barcode_five, barcode6, reseg, img_num);

			}



			cout << endl;
		}
		else    //////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置



			skip_quiet_zone(flip_img, cur); //略過空白區
			read_lguard(flip_img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(flip_img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(flip_img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(flip_img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}

					cout << "檢查碼為正確" << endl;
					four_success += 1;
					success += 1;
					onltsuc1 = 1;
					four_read = 1;

					bar_or_digits = 1;

					reseg = 1;

					check(check_num, img_num);

				}
				else
				{
					cout << endl;
					cout << "第4階段 解碼錯誤" << endl;
					cout << "換銳化 區域OTSU " << endl;
					five_read_barcode(barcode_five, barcode6, reseg, img_num);

				}



				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "第4階段 解碼錯誤" << endl;
				cout << "換銳化 區域OTSU " << endl;
				five_read_barcode(barcode_five, barcode6, reseg, img_num);

			}



			cout << endl;

		}

	}

	//imshow("flip_image", flip_img);

}



/////////////////////////////////////////////////////////////////// 第四階段    第三次掃描

void four_read_barcode3(MatU& img, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;

	MatU barcode_five = barcode5;


	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, (size / 4));  ///掃描位置
							   /*bitwise_not(img, img);
							   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							   //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}


			cout << "檢查碼為正確" << endl;
			three_success += 1;
			success += 1;
			onltsuc1 = 1;
			four_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第四次掃描" << endl;

			four_read_barcode4(img, barcode_five, barcode6, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第四次掃描" << endl;

		four_read_barcode4(img, barcode_five, barcode6, reseg, img_num);

	}



}


///////////////////////////////////////////////////////////////////////// 第四階段    第二次掃描
void  four_read_barcode2(MatU& img, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;


	MatU barcode_five = barcode5;

	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, ((size / 4) * 3));  ///掃描位置
									 /*bitwise_not(img, img);
									 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}


			cout << "檢查碼為正確" << endl;
			two_success += 1;
			success += 1;
			onltsuc1 = 1;
			four_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第三次掃描" << endl;

			four_read_barcode3(img, barcode_five, barcode6, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第三次掃描" << endl;

		four_read_barcode3(img, barcode_five, barcode6, reseg, img_num);
	}

}


//////////////////////////////////////////////////////////////////////////第四階段   第一次掃描
void  four_read_barcode(MatU& img, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;

	MatU barcode_five = barcode5;
	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, size / 2);  ///掃描位置
							 /*bitwise_not(img, img);
							 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}


			cout << "檢查碼為正確" << endl;
			one_success += 1;
			success += 1;
			onltsuc1 = 1;
			four_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第二次掃描" << endl;
			four_read_barcode2(img, barcode_five, barcode6, reseg, img_num);
		}
		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置
										 /*bitwise_not(img, img);
										 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

										 //if (img(cur) != SPACE) return;


		skip_quiet_zone(img, cur); //略過空白區
		read_lguard(img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{
				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}


				cout << "檢查碼為正確" << endl;
				two_success += 1;
				success += 1;
				onltsuc1 = 1;
				four_read = 1;

				bar_or_digits = 1;

				reseg = 1;


				check(check_num, img_num);
			}
			else
			{
				cout << endl;
				cout << "第三次掃描" << endl;

				four_read_barcode3(img, barcode_five, barcode6, reseg, img_num);

			}
			cout << endl;
		}
		else  //////////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置
									   /*bitwise_not(img, img);
									   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									   //if (img(cur) != SPACE) return;


			skip_quiet_zone(img, cur); //略過空白區
			read_lguard(img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}


					cout << "檢查碼為正確" << endl;
					three_success += 1;
					success += 1;
					onltsuc1 = 1;
					four_read = 1;

					bar_or_digits = 1;

					reseg = 1;

					check(check_num, img_num);
				}
				else
				{
					cout << endl;
					cout << "第四次掃描" << endl;

					four_read_barcode4(img, barcode_five, barcode6, reseg, img_num);

				}
				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "第四次掃描" << endl;

				four_read_barcode4(img, barcode_five, barcode6, reseg, img_num);

			}
		}
	}
	//waitKey();
}


///////////////////////////////////////////////////////////////////////////




///////////////////////////////////////////////////////////////////////// 第三階段  
void third_read_barcode4(MatU& img, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;

	MatU barcode_four = barcode4;
	MatU barcode_five = barcode5;
	MatU flip_img;



	flip(img, flip_img, 1);///水平翻轉

	int size = flip_img.rows;

	Point cur(0, (size / 2));  ///掃描位置



	skip_quiet_zone(flip_img, cur); //略過空白區
	read_lguard(flip_img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(flip_img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(flip_img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(flip_img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}


			cout << "檢查碼為正確" << endl;
			four_success += 1;
			success += 1;
			onltsuc1 = 1;
			three_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);

		}
		else
		{
			cout << endl;
			cout << "第3階段 解碼錯誤" << endl;
			cout << "換全域 adaptive " << endl;
			four_read_barcode(barcode_four, barcode_five, barcode6, reseg, img_num);

		}

		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置



		skip_quiet_zone(flip_img, cur); //略過空白區
		read_lguard(flip_img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(flip_img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(flip_img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(flip_img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{
				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}

				cout << "檢查碼為正確" << endl;
				four_success += 1;
				success += 1;
				onltsuc1 = 1;
				three_read = 1;

				bar_or_digits = 1;

				reseg = 1;

				check(check_num, img_num);

			}
			else
			{
				cout << endl;
				cout << "第3階段 解碼錯誤" << endl;
				cout << "換全域 adaptive " << endl;
				four_read_barcode(barcode_four, barcode_five, barcode6, reseg, img_num);

			}

			cout << endl;
		}
		else    //////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置



			skip_quiet_zone(flip_img, cur); //略過空白區
			read_lguard(flip_img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(flip_img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(flip_img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(flip_img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}



					cout << "檢查碼為正確" << endl;
					four_success += 1;
					success += 1;
					onltsuc1 = 1;
					three_read = 1;

					bar_or_digits = 1;

					reseg = 1;

					check(check_num, img_num);

				}
				else
				{
					cout << endl;
					cout << "第3階段 解碼錯誤" << endl;
					cout << "換全域 adaptive " << endl;
					four_read_barcode(barcode_four, barcode_five, barcode6, reseg, img_num);

				}
				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "第3階段 解碼錯誤" << endl;
				cout << "換全域 adaptive " << endl;
				four_read_barcode(barcode_four, barcode_five, barcode6, reseg, img_num);

			}

		}

	}

	//imshow("flip_image", flip_img);

}



/////////////////////////////////////////////////////////////////// 第三階段    第三次掃描

void third_read_barcode3(MatU& img, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;


	MatU barcode_four = barcode4;
	MatU barcode_five = barcode5;
	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, (size / 4));  ///掃描位置
							   /*bitwise_not(img, img);
							   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							   //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}



			cout << "檢查碼為正確" << endl;
			three_success += 1;
			success += 1;
			onltsuc1 = 1;
			three_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第四次掃描" << endl;

			third_read_barcode4(img, barcode_four, barcode_five, barcode6, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第四次掃描" << endl;

		third_read_barcode4(img, barcode_four, barcode_five, barcode6, reseg, img_num);

	}



}


///////////////////////////////////////////////////////////////////////// 第三階段    第二次掃描
void  third_read_barcode2(MatU& img, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;


	MatU barcode_four = barcode4;
	MatU barcode_five = barcode5;
	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, ((size / 4) * 3));  ///掃描位置
									 /*bitwise_not(img, img);
									 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{

			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}


			cout << "檢查碼為正確" << endl;
			two_success += 1;
			success += 1;
			onltsuc1 = 1;
			three_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第三次掃描" << endl;

			third_read_barcode3(img, barcode_four, barcode_five, barcode6, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第三次掃描" << endl;

		third_read_barcode3(img, barcode_four, barcode_five, barcode6, reseg, img_num);
	}

}


//////////////////////////////////////////////////////////////////////////第三階段   第一次掃描
void  third_read_barcode(MatU& img, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;


	MatU barcode_four = barcode4;
	MatU barcode_five = barcode5;

	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, size / 2);  ///掃描位置
							 /*bitwise_not(img, img);
							 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{

			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}


			cout << "檢查碼為正確" << endl;
			one_success += 1;
			success += 1;
			onltsuc1 = 1;
			three_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第二次掃描" << endl;
			third_read_barcode2(img, barcode_four, barcode_five, barcode6, reseg, img_num);
		}
		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置
										 /*bitwise_not(img, img);
										 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

										 //if (img(cur) != SPACE) return;


		skip_quiet_zone(img, cur); //略過空白區
		read_lguard(img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{
				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}


				cout << "檢查碼為正確" << endl;
				two_success += 1;
				success += 1;
				onltsuc1 = 1;

				bar_or_digits = 1;

				reseg = 1;

				check(check_num, img_num);
			}
			else
			{
				cout << endl;
				cout << "第三次掃描" << endl;

				third_read_barcode3(img, barcode_four, barcode_five, barcode6, reseg, img_num);

			}
			cout << endl;
		}
		else  //////////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置
									   /*bitwise_not(img, img);
									   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									   //if (img(cur) != SPACE) return;


			skip_quiet_zone(img, cur); //略過空白區
			read_lguard(img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}


					cout << "檢查碼為正確" << endl;
					three_success += 1;
					success += 1;
					onltsuc1 = 1;
					three_read = 1;

					bar_or_digits = 1;

					reseg = 1;

					check(check_num, img_num);
				}
				else
				{
					cout << endl;
					cout << "第四次掃描" << endl;

					third_read_barcode4(img, barcode_four, barcode_five, barcode6, reseg, img_num);

				}
				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "第四次掃描" << endl;

				third_read_barcode4(img, barcode_four, barcode_five, barcode6, reseg, img_num);

			}
		}
	}
	//waitKey();
}


///////////////////////////////////////////////////////////////////////////




///////////////////////////////////////////////////////////////////////// 第二階段  
void sec_read_barcode4(MatU& img, MatU& barcode3, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num) {
	z = 0;

	MatU barcode_three = barcode3;
	MatU barcode_four = barcode4;
	MatU flip_img;
	MatU barcode_five = barcode5;


	flip(img, flip_img, 1);///水平翻轉

	int size = flip_img.rows;

	Point cur(0, (size / 2));  ///掃描位置



	skip_quiet_zone(flip_img, cur); //略過空白區
	read_lguard(flip_img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(flip_img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(flip_img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(flip_img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			four_success += 1;
			success += 1;
			onltsuc1 = 1;
			two_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第2階段 解碼錯誤" << endl;
			cout << "換全域 OTSU " << endl;
			//third_read_barcode(barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);
			four_read_barcode(barcode_four, barcode_five, barcode6, reseg, img_num);

		}


		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置



		skip_quiet_zone(flip_img, cur); //略過空白區
		read_lguard(flip_img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(flip_img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(flip_img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(flip_img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{
				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}

				cout << "檢查碼為正確" << endl;
				four_success += 1;
				success += 1;
				onltsuc1 = 1;
				two_read = 1;

				bar_or_digits = 1;

				reseg = 1;

				check(check_num, img_num);

			}
			else
			{
				cout << endl;
				cout << "第2階段 解碼錯誤" << endl;
				cout << "換全域 OTSU " << endl;
				//third_read_barcode(barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);
				four_read_barcode(barcode_four, barcode_five, barcode6, reseg, img_num);
			}

			cout << endl;
		}
		else    //////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置



			skip_quiet_zone(flip_img, cur); //略過空白區
			read_lguard(flip_img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(flip_img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(flip_img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(flip_img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}


					cout << "檢查碼為正確" << endl;
					four_success += 1;
					success += 1;
					onltsuc1 = 1;
					two_read = 1;

					bar_or_digits = 1;

					reseg = 1;

					check(check_num, img_num);

				}
				else
				{
					cout << endl;
					cout << "第2階段 解碼錯誤" << endl;
					cout << "換全域 OTSU " << endl;
					//third_read_barcode(barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);
					four_read_barcode(barcode_four, barcode_five, barcode6, reseg, img_num);
				}
				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "第2階段 解碼錯誤" << endl;
				cout << "換全域 OTSU " << endl;
				//third_read_barcode(barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);
				four_read_barcode(barcode_four, barcode_five, barcode6, reseg, img_num);
			}

		}

	}

	//imshow("flip_image", flip_img);

}



/////////////////////////////////////////////////////////////////// 第二階段  第三次掃描

void  sec_read_barcode3(MatU& img, MatU& barcode3, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num) {
	z = 0;

	MatU barcode_three = barcode3;
	MatU barcode_four = barcode4;
	MatU barcode_five = barcode5;

	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, (size / 4));  ///掃描位置
							   /*bitwise_not(img, img);
							   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							   //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}


			cout << "檢查碼為正確" << endl;
			three_success += 1;
			success += 1;
			onltsuc1 = 1;
			two_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第四次掃描" << endl;

			sec_read_barcode4(img, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第四次掃描" << endl;

		sec_read_barcode4(img, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

	}



}


///////////////////////////////////////////////////////////////////////// 第二階段  第二次掃描
void  sec_read_barcode2(MatU& img, MatU& barcode3, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num) {
	z = 0;

	MatU barcode_three = barcode3;
	MatU barcode_four = barcode4;
	MatU barcode_five = barcode5;

	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, ((size / 4) * 3));  ///掃描位置
									 /*bitwise_not(img, img);
									 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			two_success += 1;
			success += 1;
			onltsuc1 = 1;
			two_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第三次掃描" << endl;

			sec_read_barcode3(img, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第三次掃描" << endl;

		sec_read_barcode3(img, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);
	}

}


//////////////////////////////////////////////////////////////////////////第二階段 第一次掃描
void sec_read_barcode(MatU& img, MatU& barcode3, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num) {
	z = 0;

	MatU barcode_three = barcode3;
	MatU barcode_four = barcode4;
	MatU barcode_five = barcode5;
	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, size / 2);  ///掃描位置
							 /*bitwise_not(img, img);
							 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			one_success += 1;
			success += 1;
			onltsuc1 = 1;
			two_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第二次掃描" << endl;
			sec_read_barcode2(img, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);
		}
		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置
										 /*bitwise_not(img, img);
										 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

										 //if (img(cur) != SPACE) return;


		skip_quiet_zone(img, cur); //略過空白區
		read_lguard(img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{
				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}

				cout << "檢查碼為正確" << endl;
				two_success += 1;
				success += 1;
				onltsuc1 = 1;
				two_read = 1;

				bar_or_digits = 1;

				reseg = 1;

				check(check_num, img_num);
			}
			else
			{
				cout << endl;
				cout << "第三次掃描" << endl;

				sec_read_barcode3(img, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

			}
			cout << endl;
		}
		else  //////////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置
									   /*bitwise_not(img, img);
									   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									   //if (img(cur) != SPACE) return;


			skip_quiet_zone(img, cur); //略過空白區
			read_lguard(img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}

					cout << "檢查碼為正確" << endl;
					three_success += 1;
					success += 1;
					onltsuc1 = 1;
					two_read = 1;

					bar_or_digits = 1;

					reseg = 1;

					check(check_num, img_num);
				}
				else
				{
					cout << endl;
					cout << "第四次掃描" << endl;

					sec_read_barcode4(img, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

				}
				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "第四次掃描" << endl;

				sec_read_barcode4(img, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

			}
		}
	}
	//waitKey();
}


///////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////水平翻轉後掃描
void read_barcode4(MatU& img, MatU& barcode2, MatU& barcode3, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;
	MatU flip_img;


	MatU barcode_two = barcode2;
	MatU barcode_three = barcode3;
	MatU barcode_four = barcode4;
	MatU barcode_five = barcode5;

	flip(img, flip_img, 1);///水平翻轉

	int size = flip_img.rows;

	Point cur(0, (size / 2));  ///掃描位置



	skip_quiet_zone(flip_img, cur); //略過空白區
	read_lguard(flip_img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(flip_img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(flip_img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(flip_img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}


			cout << "檢查碼為正確" << endl;
			four_success += 1;
			success += 1;
			onltsuc1 = 1;
			one_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);

		}
		else
		{
			cout << endl;
			cout << "第一階段 解碼錯誤" << endl;
			cout << "換全域 Bernsen" << endl;
			sec_read_barcode(barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

		}


		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置



		skip_quiet_zone(flip_img, cur); //略過空白區
		read_lguard(flip_img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(flip_img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(flip_img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(flip_img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{
				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}


				cout << "檢查碼為正確" << endl;
				four_success += 1;
				success += 1;
				onltsuc1 = 1;
				one_read = 1;

				bar_or_digits = 1;

				reseg = 1;

				check(check_num, img_num);

			}
			else
			{
				cout << endl;
				cout << "第一階段 解碼錯誤" << endl;
				cout << "換全域 Bernsen" << endl;
				sec_read_barcode(barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

			}

			cout << endl;
		}
		else    //////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置



			skip_quiet_zone(flip_img, cur); //略過空白區
			read_lguard(flip_img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(flip_img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(flip_img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(flip_img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}

					cout << "檢查碼為正確" << endl;
					four_success += 1;
					success += 1;
					onltsuc1 = 1;
					one_read = 1;

					bar_or_digits = 1;

					reseg = 1;

					check(check_num, img_num);

				}
				else
				{
					cout << endl;
					cout << "第一階段 解碼錯誤" << endl;
					cout << "換全域 Bernsen" << endl;
					sec_read_barcode(barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

				}
				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "第一階段 解碼錯誤" << endl;
				cout << "換全域 Bernsen" << endl;
				sec_read_barcode(barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

			}

		}

	}

	//imshow("flip_image", flip_img);

}



///////////////////////////////////////////////////////////////////第三次掃描

void read_barcode3(MatU& img, MatU& barcode2, MatU& barcode3, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;
	MatU barcode_two = barcode2;
	MatU barcode_three = barcode3;
	MatU barcode_four = barcode4;
	MatU barcode_five = barcode5;
	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, (size / 4));  ///掃描位置
							   /*bitwise_not(img, img);
							   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							   //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}

			cout << "檢查碼為正確" << endl;
			three_success += 1;
			success += 1;
			onltsuc1 = 1;
			one_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第四次掃描" << endl;

			read_barcode4(img, barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第四次掃描" << endl;

		read_barcode4(img, barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

	}



}


/////////////////////////////////////////////////////////////////////////第二次掃描
void read_barcode2(MatU& img, MatU& barcode2, MatU& barcode3, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num)
{
	z = 0;
	MatU barcode_two = barcode2;
	MatU barcode_three = barcode3;
	MatU barcode_four = barcode4;
	MatU barcode_five = barcode5;
	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, ((size / 4) * 3));  ///掃描位置
									 /*bitwise_not(img, img);
									 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}


			cout << "檢查碼為正確" << endl;
			two_success += 1;
			success += 1;
			onltsuc1 = 1;
			one_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第三次掃描" << endl;

			read_barcode3(img, barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

		}
		cout << endl;
	}
	else
	{
		cout << endl;
		cout << "第三次掃描" << endl;

		read_barcode3(img, barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);
	}

}


//////////////////////////////////////////////////////////////////////////第一次掃描
void read_barcode(MatU& img, MatU& barcode2, MatU& barcode3, MatU& barcode4, MatU& barcode5, MatU& barcode6, int& reseg, int& img_num) {
	z = 0;
	MatU barcode_two = barcode2;
	MatU barcode_three = barcode3;
	MatU barcode_four = barcode4;
	MatU barcode_five = barcode5;

	int size = img.rows;
	//int scanh = (size.height / 2);
	Point cur(0, size / 2);  ///掃描位置
							 /*bitwise_not(img, img);
							 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

							 //if (img(cur) != SPACE) return;


	skip_quiet_zone(img, cur); //略過空白區
	read_lguard(img, cur);
	vector<int> digits;
	int yorn = 1;

	for (int i = 0; i < 6; i++) {          //////讀左資料區
		int d = read_digit(img, cur, LEFT);
		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
	}

	skip_mguard(img, cur);   ////略過中線

	int iGetCount = 0;

	for (int i = 0; i < 6; i++) {       //////讀右資料區

		int d = read_digit(img, cur, RIGHT);


		if (d == ERROR_CODE)
		{
			yorn = 0;
		}
		else
			digits.push_back(d);
		iGetCount++;
	}

	if (yorn == 1)
	{

		int front = get_front();
		cout << front << " ";

		for (int i = 0; i < 12; i++)
			cout << digits[i] << " ";
		cout << endl;


		////////////////////////////////////////////////////////////檢查碼驗證
		int t1, t2, t3, t4;
		t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

		t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

		t3 = (t1 + t2) % 10;


		t4 = (10 - t3);

		if (t4 == 10)
		{
			t4 = 0;
		}

		cout << endl;

		cout << "check digits: " << t4 << " " << endl;

		if (t4 == digits[11])
		{
			check_num[0] = front;

			for (int j = 1; j < 13; j++)
			{
				check_num[j] = digits[j - 1];
			}


			cout << "檢查碼為正確" << endl;
			one_success += 1;
			success += 1;
			onltsuc1 = 1;
			one_read = 1;

			bar_or_digits = 1;

			reseg = 1;

			check(check_num, img_num);
		}
		else
		{
			cout << endl;
			cout << "第二次掃描" << endl;
			read_barcode2(img, barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);
		}
		cout << endl;
	}
	else  //////////////////////////////////////////////////////////////////////掃描影像高度的3/4位置
	{
		Point cur(0, ((size / 4) * 3));  ///掃描位置
										 /*bitwise_not(img, img);
										 threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

										 //if (img(cur) != SPACE) return;


		skip_quiet_zone(img, cur); //略過空白區
		read_lguard(img, cur);
		vector<int> digits;
		int yorn = 1;

		for (int i = 0; i < 6; i++) {          //////讀左資料區
			int d = read_digit(img, cur, LEFT);
			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
		}

		skip_mguard(img, cur);   ////略過中線

		int iGetCount = 0;

		for (int i = 0; i < 6; i++) {       //////讀右資料區

			int d = read_digit(img, cur, RIGHT);


			if (d == ERROR_CODE)
			{
				yorn = 0;
			}
			else
				digits.push_back(d);
			iGetCount++;
		}

		if (yorn == 1)
		{

			int front = get_front();
			cout << front << " ";

			for (int i = 0; i < 12; i++)
				cout << digits[i] << " ";
			cout << endl;


			////////////////////////////////////////////////////////////檢查碼驗證
			int t1, t2, t3, t4;
			t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

			t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

			t3 = (t1 + t2) % 10;


			t4 = (10 - t3);

			if (t4 == 10)
			{
				t4 = 0;
			}

			cout << endl;

			cout << "check digits: " << t4 << " " << endl;

			if (t4 == digits[11])
			{
				check_num[0] = front;

				for (int j = 1; j < 13; j++)
				{
					check_num[j] = digits[j - 1];
				}

				cout << "檢查碼為正確" << endl;
				two_success += 1;
				success += 1;
				onltsuc1 = 1;
				one_read = 1;

				bar_or_digits = 1;

				reseg = 1;

				check(check_num, img_num);
			}
			else
			{
				cout << endl;
				cout << "第三次掃描" << endl;

				read_barcode3(img, barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

			}
			cout << endl;
		}
		else  //////////////////////////////////////////////////////////////////////掃描影像高度的1/4位置
		{
			Point cur(0, (size / 4));  ///掃描位置
									   /*bitwise_not(img, img);
									   threshold(img, img, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

									   //if (img(cur) != SPACE) return;


			skip_quiet_zone(img, cur); //略過空白區
			read_lguard(img, cur);
			vector<int> digits;
			int yorn = 1;

			for (int i = 0; i < 6; i++) {          //////讀左資料區
				int d = read_digit(img, cur, LEFT);
				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
			}

			skip_mguard(img, cur);   ////略過中線

			int iGetCount = 0;

			for (int i = 0; i < 6; i++) {       //////讀右資料區

				int d = read_digit(img, cur, RIGHT);


				if (d == ERROR_CODE)
				{
					yorn = 0;
				}
				else
					digits.push_back(d);
				iGetCount++;
			}

			if (yorn == 1)
			{

				int front = get_front();
				cout << front << " ";

				for (int i = 0; i < 12; i++)
					cout << digits[i] << " ";
				cout << endl;


				////////////////////////////////////////////////////////////檢查碼驗證
				int t1, t2, t3, t4;
				t1 = (digits[0] + digits[2] + digits[4] + digits[6] + digits[8] + digits[10]) * 3;

				t2 = (front + digits[1] + digits[3] + digits[5] + digits[7] + digits[9]);

				t3 = (t1 + t2) % 10;


				t4 = (10 - t3);

				if (t4 == 10)
				{
					t4 = 0;
				}

				cout << endl;

				cout << "check digits: " << t4 << " " << endl;

				if (t4 == digits[11])
				{
					check_num[0] = front;

					for (int j = 1; j < 13; j++)
					{
						check_num[j] = digits[j - 1];
					}

					cout << "檢查碼為正確" << endl;
					three_success += 1;
					success += 1;
					onltsuc1 = 1;
					one_read = 1;

					bar_or_digits = 1;

					reseg = 1;

					check(check_num, img_num);
				}
				else
				{
					cout << endl;
					cout << "第四次掃描" << endl;

					read_barcode4(img, barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

				}
				cout << endl;
			}
			else
			{
				cout << endl;
				cout << "第四次掃描" << endl;

				read_barcode4(img, barcode_two, barcode_three, barcode_four, barcode_five, barcode6, reseg, img_num);

			}
		}
	}
	//waitKey();
}

//////////////////////////////////////////

Mat thresh_bernsen(Mat& gray, int ksize, int contrast_limit)
{
	Mat ret = Mat::zeros(gray.size(), gray.type());
	for (int i = 0; i < gray.cols; i++)
	{
		for (int j = 0; j < gray.rows; j++)
		{
			double mn = 999, mx = 0;
			int ti = 0, tj = 0;
			int tlx = i - ksize / 2;
			int tly = j - ksize / 2;
			int brx = i + ksize / 2;
			int bry = j + ksize / 2;
			if (tlx < 0) tlx = 0;
			if (tly < 0) tly = 0;
			if (brx >= gray.cols) brx = gray.cols - 1;
			if (bry >= gray.rows) bry = gray.rows - 1;

			minMaxIdx(gray(Rect(Point(tlx, tly), Point(brx, bry))), &mn, &mx, 0, 0);
			/* this does the above
			for(int ik=-ksize/2;ik<=ksize/2;ik++)
			{
			for(int jk=-ksize/2;jk<=ksize/2;jk++)
			{
			ti=i+ik;
			tj=j+jk;
			if(ti>0 && ti<gray.cols && tj>0 && tj<gray.rows)
			{
			uchar pix = gray.at<uchar>(tj,ti);
			if(pix<mn) mn=pix;
			if(pix>mx) mx=pix;
			}
			}
			}*/
			int median = 0.5 * (mn + mx);
			if (median < contrast_limit)
			{
				ret.at<uchar>(j, i) = 0;
			}
			else
			{
				uchar pix = gray.at<uchar>(j, i);
				ret.at<uchar>(j, i) = pix > median ? 255 : 0;
			}
		}
	}
	return ret;
}

/////////////////////////////////

void knn(Mat & img)
{

	std::vector<ContourWithData> allContoursWithData;           // declare empty vectors,
	std::vector<ContourWithData> validContoursWithData;         // we will fill these shortly

																// read in training classifications ///////////////////////////////////////////////////

	cv::Mat matClassificationInts;      // we will read the classification numbers into this variable as though it is a vector

	cv::FileStorage fsClassifications("501_new_classifications.xml", cv::FileStorage::READ);        // open the classifications file



	fsClassifications["classifications"] >> matClassificationInts;      // read classifications section into Mat classifications variable
	fsClassifications.release();                                        // close the classifications file

																		// read in training images ////////////////////////////////////////////////////////////

	cv::Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

	cv::FileStorage fsTrainingImages("501_new_images.xml", cv::FileStorage::READ);          // open the training images file



	fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
	fsTrainingImages.release();                                                 // close the traning images file

																				// train //////////////////////////////////////////////////////////////////////////////

	cv::Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());            // instantiate the KNN object

																				// finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
																				// even though in reality they are multiple images / numbers
	kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts);



	std::string strFinalString;
	cv::Mat matROIResized;

	cv::resize(img, matROIResized, cv::Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));     // resize image, this will be more consistent for recognition and storage

	cv::Mat matROIFloat;
	matROIResized.convertTo(matROIFloat, CV_32FC1);             // convert Mat to float, necessary for call to find_nearest

	cv::Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);

	cv::Mat matCurrentChar(0, 0, CV_32F);

	kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);     // finally we can call find_nearest !!!

	float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);

	strFinalString = strFinalString + char(int(fltCurrentChar));        // append current char to full string


																		//std::cout << "numbers read = " << strFinalString << "\n";       // show the full string


	digitresult[0].push_back(strFinalString);


}


void svm(Mat & img, int& svm_digit)
{
	Mat testImages, img2;

	Ptr<SVM> svm2 = Algorithm::load<SVM>("SVM.xml");

	if (img.rows > 0 && img.cols > 0)
	{
		cv::resize(img, img, cv::Size(12, 18));

		if (img.isContinuous()) {
			img2 = img.reshape(1, 1);

			Mat(img2).copyTo(testImages);

			testImages.convertTo(testImages, CV_32FC1);
			svm_digit = svm2->predict(testImages);
		}
		else
		{
			svm_digit = -999;
		}

	}
	else
	{
		svm_digit = -999;
	}
}


void seg_barcode2(Mat& Max_Location)
{
	Size size = Max_Location.size();
	int h1 = (size.height / 2);
	int h2 = (size.height / 2);
	int w = size.width;
	Mat digitImage = Max_Location(Rect(0, h1, w, h2));






	///////////////////////////////////////////////////////////////////////////////////////直方圖
	Mat Horizontalprojection;

	Horizontalprojection.create(digitImage.rows, digitImage.cols, CV_8U);
	if ((digitImage.rows) != 0 || (digitImage.cols) != 0)
	{

		for (int r = 0; r < digitImage.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < digitImage.cols; c++)
			{
				if (digitImage.at<uchar>(r, c) == 0) Horizontalprojection.at<uchar>(r, a++) = 0;


			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////直方圖切割
		Size dImgsize = digitImage.size();
		int neww = (digitImage.cols);
		int newh = (digitImage.rows);


		int maxblack = 0, maxblack1 = 0, cutline = 0;
		std::vector<int> Hiss(digitImage.rows);
		for (int i = 0; i < Horizontalprojection.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection.cols; j++)
			{
				if (Horizontalprojection.at<uchar>(i, j) == 0)
				{
					Hiss[i]++;
				}
			}




		}


		for (size_t i = 0; i < (Hiss.size()*0.8); i++)
		{
			//std::cout << Hiss[i] << endl;
			if (maxblack < Hiss[i])
			{
				maxblack = Hiss[i];

				cutline = i;



			}
		}




		int hh = (newh - cutline);


		Mat digitImage3 = digitImage(Rect(0, newh / 2, neww, newh / 2));/////////////////////////數字影像

																		//cutline = cutline;// Horizontalprojection.at<uchar>(maxblack1, )

																		//Mat digitImage2 = digitImage(Rect(0, cutline, dImgsize.width, newh));
																		//cout << "cutline: " << cutline << endl;
		cout << "newh: " << newh << endl;
		cout << "neww: " << neww << endl;
		///////////////////////////////////////////////////////////////////////////////////////


		////////////////////////////////////////////////////////////////////////////////去護線

		cv::Mat labelImage3;
		cv::Mat stats3, centroids3;


		//Mat digits_kernel = getStructuringElement(MORPH_RECT, Size(2, 1));/////////////////////////////////////////////數字修補
		//morphologyEx(digitImage3, digitImage3, MORPH_OPEN, digits_kernel);
		//dilate(digitImage3, digitImage3, digits_kernel);


		int nLabels3 = cv::connectedComponentsWithStats(digitImage3, labelImage3, stats3, centroids3, 8, CV_32S);/////八連通

		std::vector<cv::Vec3b> colors3(nLabels3);
		colors3[0] = cv::Vec3b(0, 0, 0);
		//std::cout << "Number of connected components = " << nLabels2 << std::endl << std::endl;

		//int newh3 = (((digitImage3.rows) * 5) / 10);                                //////////////////// 護線長度過濾

		for (int label = 1; label < nLabels3; ++label) {
			colors3[label] = cv::Vec3b(255, 255, 255);
			//std::cout << "Component " << label << std::endl;
			//cout << "CC_STAT_TOP    = " << stats3.at<int>(label, cv::CC_STAT_TOP) << std::endl;
			//std::cout << "CC_STAT_HEIGHT = " << stats3.at<int>(label, cv::CC_STAT_HEIGHT) << std::endl;

			if (stats3.at<int>(label, cv::CC_STAT_TOP) == 0 || stats3.at<int>(label, cv::CC_STAT_AREA)<5)                   /////////////////////////////////////////////   過濾護線 與上方相連的特徵
			{
				colors3[label] = cv::Vec3b(0, 0, 0);
			}

		}

		cv::Mat dd(digitImage3.size(), CV_8UC3);////上色
		for (int r = 0; r < dd.rows; ++r) {
			for (int c = 0; c < dd.cols; ++c) {
				int label = labelImage3.at<int>(r, c);
				cv::Vec3b &pixel = dd.at<cv::Vec3b>(r, c);
				pixel = colors3[label];
			}
		}








		//imshow("Horizontalprojection", Horizontalprojection);
		imwrite("Horizontalprojection.jpg", Horizontalprojection);

		//imshow("digitImage3", digitImage3);
		//imshow("onlydigitImage3.", dd);
		imwrite("onlydigitImage3.jpg", dd);
	}
}


void seg_barcode(Mat& Max_Location)
{
	Size size = Max_Location.size();
	int h1 = (size.height / 2);
	int h2 = (size.height / 2);
	int w = size.width;
	Mat digitImage = Max_Location(Rect(0, h1, w, h2));






	///////////////////////////////////////////////////////////////////////////////////////直方圖
	Mat Horizontalprojection;

	Horizontalprojection.create(digitImage.rows, digitImage.cols, CV_8U);
	if ((digitImage.rows) != 0 || (digitImage.cols) != 0)
	{

		for (int r = 0; r < digitImage.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < digitImage.cols; c++)
			{
				if (digitImage.at<uchar>(r, c) == 0) Horizontalprojection.at<uchar>(r, a++) = 0;


			}
		}
		///////////////////////////////////////////////////////////////////////////////////////////////直方圖切割
		Size dImgsize = digitImage.size();
		int neww = (digitImage.cols);
		int newh = (digitImage.rows);


		int maxblack = 0, maxblack1 = 0, cutline = 0;
		std::vector<int> Hiss(digitImage.rows);
		for (int i = 0; i < Horizontalprojection.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection.cols; j++)
			{
				if (Horizontalprojection.at<uchar>(i, j) == 0)
				{
					Hiss[i]++;
				}
			}




		}


		for (size_t i = 0; i < (Hiss.size()*0.8); i++)
		{
			//std::cout << Hiss[i] << endl;
			if (maxblack < Hiss[i])
			{
				maxblack = Hiss[i];

				cutline = i;



			}
		}




		int hh = (newh - cutline);


		Mat digitImage3 = digitImage(Rect(0, newh / 2, neww, newh / 2));/////////////////////////數字影像

																		//cutline = cutline;// Horizontalprojection.at<uchar>(maxblack1, )

																		//Mat digitImage2 = digitImage(Rect(0, cutline, dImgsize.width, newh));
																		//cout << "cutline: " << cutline << endl;
		cout << "newh: " << newh << endl;
		cout << "neww: " << neww << endl;
		///////////////////////////////////////////////////////////////////////////////////////


		////////////////////////////////////////////////////////////////////////////////去護線

		cv::Mat labelImage3;
		cv::Mat stats3, centroids3;


		//Mat digits_kernel = getStructuringElement(MORPH_RECT, Size(2, 1));/////////////////////////////////////////////數字修補
		//morphologyEx(digitImage3, digitImage3, MORPH_OPEN, digits_kernel);
		//dilate(digitImage3, digitImage3, digits_kernel);


		int nLabels3 = cv::connectedComponentsWithStats(digitImage3, labelImage3, stats3, centroids3, 8, CV_32S);/////八連通

		std::vector<cv::Vec3b> colors3(nLabels3);
		colors3[0] = cv::Vec3b(0, 0, 0);
		//std::cout << "Number of connected components = " << nLabels2 << std::endl << std::endl;

		//int newh3 = (((digitImage3.rows) * 5) / 10);                                //////////////////// 護線長度過濾

		for (int label = 1; label < nLabels3; ++label) {
			colors3[label] = cv::Vec3b(255, 255, 255);
			//std::cout << "Component " << label << std::endl;
			//cout << "CC_STAT_TOP    = " << stats3.at<int>(label, cv::CC_STAT_TOP) << std::endl;
			//std::cout << "CC_STAT_HEIGHT = " << stats3.at<int>(label, cv::CC_STAT_HEIGHT) << std::endl;

			if (stats3.at<int>(label, cv::CC_STAT_TOP) == 0)                   /////////////////////////////////////////////   過濾護線 與上方相連的特徵
			{
				colors3[label] = cv::Vec3b(0, 0, 0);
			}

		}

		cv::Mat dd(digitImage3.size(), CV_8UC3);////上色
		for (int r = 0; r < dd.rows; ++r) {
			for (int c = 0; c < dd.cols; ++c) {
				int label = labelImage3.at<int>(r, c);
				cv::Vec3b &pixel = dd.at<cv::Vec3b>(r, c);
				pixel = colors3[label];
			}
		}








		//imshow("Horizontalprojection", Horizontalprojection);
		imwrite("Horizontalprojection.jpg", Horizontalprojection);

		//imshow("digitImage3", digitImage3);
		//imshow("onlydigitImage3.", dd);
		imwrite("onlydigitImage3.jpg", dd);
	}
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////

void svm_four_digitsort(Mat & img, int& img_num)
{

	Mat new_seg_barcod = img;

	seg_barcode2(new_seg_barcod);

	/////////////////////////////////////////////////////////////////////////////////////////數字分割


	cv::Mat dig = cv::imread("onlydigitImage3.jpg", CV_LOAD_IMAGE_GRAYSCALE);



	threshold(dig, dig, 150, 255, THRESH_BINARY | THRESH_OTSU);


	cout << "使用 全域OTSU 影像 辨識數字" << endl;


	int again_seg = 0;

	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;



	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(dig, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}

	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, row_record[0], neww, (dig.rows) - row_record[0]));


			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				cout << endl;
				int numberresult[13];

				svm(projection_d1, svm_digit);
				numberresult[0] = svm_digit;
				svm(projection_d2, svm_digit);
				numberresult[1] = svm_digit;
				svm(projection_d3, svm_digit);
				numberresult[2] = svm_digit;
				svm(projection_d4, svm_digit);
				numberresult[3] = svm_digit;
				svm(projection_d5, svm_digit);
				numberresult[4] = svm_digit;
				svm(projection_d6, svm_digit);
				numberresult[5] = svm_digit;
				svm(projection_d7, svm_digit);
				numberresult[6] = svm_digit;
				svm(projection_d8, svm_digit);
				numberresult[7] = svm_digit;
				svm(projection_d9, svm_digit);
				numberresult[8] = svm_digit;
				svm(projection_d10, svm_digit);
				numberresult[9] = svm_digit;
				svm(projection_d11, svm_digit);
				numberresult[10] = svm_digit;
				svm(projection_d12, svm_digit);
				numberresult[11] = svm_digit;
				svm(projection_d13, svm_digit);
				numberresult[12] = svm_digit;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";





				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{

					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);


				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = dig(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = dig(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = dig(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = dig(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = dig(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = dig(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = dig(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = dig(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = dig(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = dig(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = dig(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = dig(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = dig(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));


		cout << endl;
		int numberresult[13];

		svm(d1, svm_digit);
		numberresult[0] = svm_digit;
		svm(d2, svm_digit);
		numberresult[1] = svm_digit;
		svm(d3, svm_digit);
		numberresult[2] = svm_digit;
		svm(d4, svm_digit);
		numberresult[3] = svm_digit;
		svm(d5, svm_digit);
		numberresult[4] = svm_digit;
		svm(d6, svm_digit);
		numberresult[5] = svm_digit;
		svm(d7, svm_digit);
		numberresult[6] = svm_digit;
		svm(d8, svm_digit);
		numberresult[7] = svm_digit;
		svm(d9, svm_digit);
		numberresult[8] = svm_digit;
		svm(d10, svm_digit);
		numberresult[9] = svm_digit;
		svm(d11, svm_digit);
		numberresult[10] = svm_digit;
		svm(d12, svm_digit);
		numberresult[11] = svm_digit;
		svm(d13, svm_digit);
		numberresult[12] = svm_digit;





		cout << endl;
		cout << endl;

		cout << "數字辨識為: ";





		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "全域OTSU 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;

			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, r_record, neww, (dig.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));



				cout << endl;
				int numberresult[13];

				svm(projection_d1, svm_digit);
				numberresult[0] = svm_digit;
				svm(projection_d2, svm_digit);
				numberresult[1] = svm_digit;
				svm(projection_d3, svm_digit);
				numberresult[2] = svm_digit;
				svm(projection_d4, svm_digit);
				numberresult[3] = svm_digit;
				svm(projection_d5, svm_digit);
				numberresult[4] = svm_digit;
				svm(projection_d6, svm_digit);
				numberresult[5] = svm_digit;
				svm(projection_d7, svm_digit);
				numberresult[6] = svm_digit;
				svm(projection_d8, svm_digit);
				numberresult[7] = svm_digit;
				svm(projection_d9, svm_digit);
				numberresult[8] = svm_digit;
				svm(projection_d10, svm_digit);
				numberresult[9] = svm_digit;
				svm(projection_d11, svm_digit);
				numberresult[10] = svm_digit;
				svm(projection_d12, svm_digit);
				numberresult[11] = svm_digit;
				svm(projection_d13, svm_digit);
				numberresult[12] = svm_digit;





				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";





				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}




	}





}

void svm_three_digitsort(Mat & img, Mat& linear_Location, int& img_num)
{

	Mat new_seg_barcod = img;

	seg_barcode(new_seg_barcod);

	/////////////////////////////////////////////////////////////////////////////////////////數字分割


	cv::Mat dig = cv::imread("onlydigitImage3.jpg", CV_LOAD_IMAGE_GRAYSCALE);



	threshold(dig, dig, 150, 255, THRESH_BINARY | THRESH_OTSU);


	cout << "使用 全域OTSU 影像 辨識數字" << endl;


	int again_seg = 0;
	int four_seg = 0;


	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;



	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(dig, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}

	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, row_record[0], neww, (dig.rows) - row_record[0]));


			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				cout << endl;
				int numberresult[13];

				svm(projection_d1, svm_digit);
				numberresult[0] = svm_digit;
				svm(projection_d2, svm_digit);
				numberresult[1] = svm_digit;
				svm(projection_d3, svm_digit);
				numberresult[2] = svm_digit;
				svm(projection_d4, svm_digit);
				numberresult[3] = svm_digit;
				svm(projection_d5, svm_digit);
				numberresult[4] = svm_digit;
				svm(projection_d6, svm_digit);
				numberresult[5] = svm_digit;
				svm(projection_d7, svm_digit);
				numberresult[6] = svm_digit;
				svm(projection_d8, svm_digit);
				numberresult[7] = svm_digit;
				svm(projection_d9, svm_digit);
				numberresult[8] = svm_digit;
				svm(projection_d10, svm_digit);
				numberresult[9] = svm_digit;
				svm(projection_d11, svm_digit);
				numberresult[10] = svm_digit;
				svm(projection_d12, svm_digit);
				numberresult[11] = svm_digit;
				svm(projection_d13, svm_digit);
				numberresult[12] = svm_digit;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";





				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{

					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					four_seg = 1;
					bar_or_digits = 2;

					check(numberresult, img_num);


				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = dig(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = dig(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = dig(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = dig(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = dig(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = dig(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = dig(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = dig(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = dig(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = dig(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = dig(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = dig(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = dig(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));


		cout << endl;
		int numberresult[13];

		svm(d1, svm_digit);
		numberresult[0] = svm_digit;
		svm(d2, svm_digit);
		numberresult[1] = svm_digit;
		svm(d3, svm_digit);
		numberresult[2] = svm_digit;
		svm(d4, svm_digit);
		numberresult[3] = svm_digit;
		svm(d5, svm_digit);
		numberresult[4] = svm_digit;
		svm(d6, svm_digit);
		numberresult[5] = svm_digit;
		svm(d7, svm_digit);
		numberresult[6] = svm_digit;
		svm(d8, svm_digit);
		numberresult[7] = svm_digit;
		svm(d9, svm_digit);
		numberresult[8] = svm_digit;
		svm(d10, svm_digit);
		numberresult[9] = svm_digit;
		svm(d11, svm_digit);
		numberresult[10] = svm_digit;
		svm(d12, svm_digit);
		numberresult[11] = svm_digit;
		svm(d13, svm_digit);
		numberresult[12] = svm_digit;





		cout << endl;
		cout << endl;

		cout << "數字辨識為: ";





		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "全域OTSU 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;

			four_seg = 1;
			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, r_record, neww, (dig.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));



				cout << endl;
				int numberresult[13];

				svm(projection_d1, svm_digit);
				numberresult[0] = svm_digit;
				svm(projection_d2, svm_digit);
				numberresult[1] = svm_digit;
				svm(projection_d3, svm_digit);
				numberresult[2] = svm_digit;
				svm(projection_d4, svm_digit);
				numberresult[3] = svm_digit;
				svm(projection_d5, svm_digit);
				numberresult[4] = svm_digit;
				svm(projection_d6, svm_digit);
				numberresult[5] = svm_digit;
				svm(projection_d7, svm_digit);
				numberresult[6] = svm_digit;
				svm(projection_d8, svm_digit);
				numberresult[7] = svm_digit;
				svm(projection_d9, svm_digit);
				numberresult[8] = svm_digit;
				svm(projection_d10, svm_digit);
				numberresult[9] = svm_digit;
				svm(projection_d11, svm_digit);
				numberresult[10] = svm_digit;
				svm(projection_d12, svm_digit);
				numberresult[11] = svm_digit;
				svm(projection_d13, svm_digit);
				numberresult[12] = svm_digit;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";





				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					four_seg = 1;
					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}




	}


	if (four_seg == 0)
	{
		svm_four_digitsort(linear_Location, img_num);
	}


}


void svm_again_digitsort(Mat & img, Mat & OTSU__Location, Mat& linear_Location, int& img_num)
{

	Mat new_seg_barcod = img;

	seg_barcode(new_seg_barcod);

	/////////////////////////////////////////////////////////////////////////////////////////數字分割


	cv::Mat dig = cv::imread("onlydigitImage3.jpg", CV_LOAD_IMAGE_GRAYSCALE);



	threshold(dig, dig, 150, 255, THRESH_BINARY | THRESH_OTSU);


	cout << "使用 bernsen 影像 辨識數字" << endl;


	int again_seg = 0;
	int three_seg = 0;

	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;



	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(dig, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}

	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, row_record[0], neww, (dig.rows) - row_record[0]));


			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				cout << endl;
				int numberresult[13];

				svm(projection_d1, svm_digit);
				numberresult[0] = svm_digit;
				svm(projection_d2, svm_digit);
				numberresult[1] = svm_digit;
				svm(projection_d3, svm_digit);
				numberresult[2] = svm_digit;
				svm(projection_d4, svm_digit);
				numberresult[3] = svm_digit;
				svm(projection_d5, svm_digit);
				numberresult[4] = svm_digit;
				svm(projection_d6, svm_digit);
				numberresult[5] = svm_digit;
				svm(projection_d7, svm_digit);
				numberresult[6] = svm_digit;
				svm(projection_d8, svm_digit);
				numberresult[7] = svm_digit;
				svm(projection_d9, svm_digit);
				numberresult[8] = svm_digit;
				svm(projection_d10, svm_digit);
				numberresult[9] = svm_digit;
				svm(projection_d11, svm_digit);
				numberresult[10] = svm_digit;
				svm(projection_d12, svm_digit);
				numberresult[11] = svm_digit;
				svm(projection_d13, svm_digit);
				numberresult[12] = svm_digit;





				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";



				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域bernsen 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					three_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = dig(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = dig(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = dig(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = dig(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = dig(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = dig(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = dig(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = dig(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = dig(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = dig(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = dig(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = dig(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = dig(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));


		cout << endl;
		int numberresult[13];

		svm(d1, svm_digit);
		numberresult[0] = svm_digit;
		svm(d2, svm_digit);
		numberresult[1] = svm_digit;
		svm(d3, svm_digit);
		numberresult[2] = svm_digit;
		svm(d4, svm_digit);
		numberresult[3] = svm_digit;
		svm(d5, svm_digit);
		numberresult[4] = svm_digit;
		svm(d6, svm_digit);
		numberresult[5] = svm_digit;
		svm(d7, svm_digit);
		numberresult[6] = svm_digit;
		svm(d8, svm_digit);
		numberresult[7] = svm_digit;
		svm(d9, svm_digit);
		numberresult[8] = svm_digit;
		svm(d10, svm_digit);
		numberresult[9] = svm_digit;
		svm(d11, svm_digit);
		numberresult[10] = svm_digit;
		svm(d12, svm_digit);
		numberresult[11] = svm_digit;
		svm(d13, svm_digit);
		numberresult[12] = svm_digit;





		cout << endl;
		cout << endl;

		cout << "數字辨識為: ";





		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "全域bernsen 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;
			three_seg = 1;

			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, r_record, neww, (dig.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));





				cout << endl;
				int numberresult[13];

				svm(projection_d1, svm_digit);
				numberresult[0] = svm_digit;
				svm(projection_d2, svm_digit);
				numberresult[1] = svm_digit;
				svm(projection_d3, svm_digit);
				numberresult[2] = svm_digit;
				svm(projection_d4, svm_digit);
				numberresult[3] = svm_digit;
				svm(projection_d5, svm_digit);
				numberresult[4] = svm_digit;
				svm(projection_d6, svm_digit);
				numberresult[5] = svm_digit;
				svm(projection_d7, svm_digit);
				numberresult[6] = svm_digit;
				svm(projection_d8, svm_digit);
				numberresult[7] = svm_digit;
				svm(projection_d9, svm_digit);
				numberresult[8] = svm_digit;
				svm(projection_d10, svm_digit);
				numberresult[9] = svm_digit;
				svm(projection_d11, svm_digit);
				numberresult[10] = svm_digit;
				svm(projection_d12, svm_digit);
				numberresult[11] = svm_digit;
				svm(projection_d13, svm_digit);
				numberresult[12] = svm_digit;





				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";



				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域bernsen 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					three_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}




	}

	if (three_seg == 0)
	{
		svm_three_digitsort(OTSU__Location, linear_Location, img_num);
	}


}

void svm_digitsort(Mat & img, Mat & bernsen_Location, Mat& OTSU__Location, Mat& linear_Location, int& img_num)
{
	int again_seg = 0;
	int bernsen_again_seg = 0;

	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;



	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(img, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}

	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(img.rows, img.cols, CV_8U);

		int neww = (img.cols);
		int newh = (img.rows);



		for (int r = 0; r < img.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < img.cols; c++)
			{
				if (img.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = img(Rect(0, row_record[0], neww, (img.rows) - row_record[0]));



			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));






				cout << endl;
				int numberresult[13];




				svm(projection_d1, svm_digit);
				numberresult[0] = svm_digit;
				svm(projection_d2, svm_digit);
				numberresult[1] = svm_digit;
				svm(projection_d3, svm_digit);
				numberresult[2] = svm_digit;
				svm(projection_d4, svm_digit);
				numberresult[3] = svm_digit;
				svm(projection_d5, svm_digit);
				numberresult[4] = svm_digit;
				svm(projection_d6, svm_digit);
				numberresult[5] = svm_digit;
				svm(projection_d7, svm_digit);
				numberresult[6] = svm_digit;
				svm(projection_d8, svm_digit);
				numberresult[7] = svm_digit;
				svm(projection_d9, svm_digit);
				numberresult[8] = svm_digit;
				svm(projection_d10, svm_digit);
				numberresult[9] = svm_digit;
				svm(projection_d11, svm_digit);
				numberresult[10] = svm_digit;
				svm(projection_d12, svm_digit);
				numberresult[11] = svm_digit;
				svm(projection_d13, svm_digit);
				numberresult[12] = svm_digit;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "區域OTSU 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					bernsen_again_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}

		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = img(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = img(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = img(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = img(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = img(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = img(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = img(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = img(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = img(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = img(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = img(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = img(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = img(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));

		cout << endl;
		int numberresult[13];

		svm(d1, svm_digit);
		numberresult[0] = svm_digit;
		svm(d2, svm_digit);
		numberresult[1] = svm_digit;
		svm(d3, svm_digit);
		numberresult[2] = svm_digit;
		svm(d4, svm_digit);
		numberresult[3] = svm_digit;
		svm(d5, svm_digit);
		numberresult[4] = svm_digit;
		svm(d6, svm_digit);
		numberresult[5] = svm_digit;
		svm(d7, svm_digit);
		numberresult[6] = svm_digit;
		svm(d8, svm_digit);
		numberresult[7] = svm_digit;
		svm(d9, svm_digit);
		numberresult[8] = svm_digit;
		svm(d10, svm_digit);
		numberresult[9] = svm_digit;
		svm(d11, svm_digit);
		numberresult[10] = svm_digit;
		svm(d12, svm_digit);
		numberresult[11] = svm_digit;
		svm(d13, svm_digit);
		numberresult[12] = svm_digit;




		cout << endl;
		cout << endl;

		cout << "數字辨識為: ";

		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "區域OTSU 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;
			bernsen_again_seg = 1;

			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(img.rows, img.cols, CV_8U);

		int neww = (img.cols);
		int newh = (img.rows);



		for (int r = 0; r < img.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < img.cols; c++)
			{
				if (img.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = img(Rect(0, r_record, neww, (img.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				cout << endl;
				int numberresult[13];

				svm(projection_d1, svm_digit);
				numberresult[0] = svm_digit;
				svm(projection_d2, svm_digit);
				numberresult[1] = svm_digit;
				svm(projection_d3, svm_digit);
				numberresult[2] = svm_digit;
				svm(projection_d4, svm_digit);
				numberresult[3] = svm_digit;
				svm(projection_d5, svm_digit);
				numberresult[4] = svm_digit;
				svm(projection_d6, svm_digit);
				numberresult[5] = svm_digit;
				svm(projection_d7, svm_digit);
				numberresult[6] = svm_digit;
				svm(projection_d8, svm_digit);
				numberresult[7] = svm_digit;
				svm(projection_d9, svm_digit);
				numberresult[8] = svm_digit;
				svm(projection_d10, svm_digit);
				numberresult[9] = svm_digit;
				svm(projection_d11, svm_digit);
				numberresult[10] = svm_digit;
				svm(projection_d12, svm_digit);
				numberresult[11] = svm_digit;
				svm(projection_d13, svm_digit);
				numberresult[12] = svm_digit;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "區域OTSU 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					bernsen_again_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);

				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}

		}




	}

	if (bernsen_again_seg == 0)
	{
		svm_again_digitsort(bernsen_Location, OTSU__Location, linear_Location, img_num);

	}

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////

void four_digitsort(Mat & img, Mat & bernsen_Location, Mat& OTSU__Location, Mat& linear_Location, int& img_num)
{

	Mat new_seg_barcod = linear_Location;

	seg_barcode2(new_seg_barcod);

	/////////////////////////////////////////////////////////////////////////////////////////數字分割


	cv::Mat dig = cv::imread("onlydigitImage3.jpg", CV_LOAD_IMAGE_GRAYSCALE);



	threshold(dig, dig, 150, 255, THRESH_BINARY | THRESH_OTSU);


	cout << "使用 全域OTSU 影像 辨識數字" << endl;


	int again_seg = 0;
	int xor = 0;

	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;



	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(dig, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}


	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, row_record[0], neww, (dig.rows) - row_record[0]));


			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				knn(projection_d1);
				knn(projection_d2);
				knn(projection_d3);
				knn(projection_d4);
				knn(projection_d5);
				knn(projection_d6);
				knn(projection_d7);
				knn(projection_d8);
				knn(projection_d9);
				knn(projection_d10);
				knn(projection_d11);
				knn(projection_d12);
				knn(projection_d13);


				cout << endl;
				cout << "數字辨識為: ";



				int numberresult[13];

				vector<string>  ::iterator iter = digitresult[0].begin();
				for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

					cout << *iter;

					int i = 0;
					convertFromString(i, *iter);     ////// string to int

					numberresult[ix] = i;



				}

				cout << endl;
				cout << endl;
				cout << "轉int後數字為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{

					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					xor = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);


				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = dig(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = dig(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = dig(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = dig(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = dig(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = dig(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = dig(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = dig(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = dig(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = dig(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = dig(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = dig(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = dig(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));

		knn(d1);
		knn(d2);
		knn(d3);
		knn(d4);
		knn(d5);
		knn(d6);
		knn(d7);
		knn(d8);
		knn(d9);
		knn(d10);
		knn(d11);
		knn(d12);
		knn(d13);

		cout << endl;
		cout << "數字辨識為: ";

		stringstream ss;

		int numberresult[13];

		vector<string>  ::iterator iter = digitresult[0].begin();
		for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

			cout << *iter;

			int i = 0;
			convertFromString(i, *iter);     ////// string to int

			numberresult[ix] = i;

			/*ss.str("");
			ss << *iter;
			ss >> numberresult[ix];*/

		}

		cout << endl;
		cout << endl;
		cout << "轉int後數字為: ";

		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "全域OTSU 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;

			xor = 1;

			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, r_record, neww, (dig.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				knn(projection_d1);
				knn(projection_d2);
				knn(projection_d3);
				knn(projection_d4);
				knn(projection_d5);
				knn(projection_d6);
				knn(projection_d7);
				knn(projection_d8);
				knn(projection_d9);
				knn(projection_d10);
				knn(projection_d11);
				knn(projection_d12);
				knn(projection_d13);


				cout << endl;
				cout << "數字辨識為: ";



				int numberresult[13];

				vector<string>  ::iterator iter = digitresult[0].begin();
				for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

					cout << *iter;

					int i = 0;
					convertFromString(i, *iter);     ////// string to int

					numberresult[ix] = i;



				}

				cout << endl;
				cout << endl;
				cout << "轉int後數字為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					xor = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}




	}

	if (xor == 0)
	{
		xor_digitsort(img, bernsen_Location, OTSU__Location, linear_Location, img_num);
	}


}


void three_digitsort(Mat & img, Mat & bernsen_Location, Mat& OTSU__Location, Mat& linear_Location, int& img_num)
{

	Mat new_seg_barcod = OTSU__Location;

	seg_barcode(new_seg_barcod);

	/////////////////////////////////////////////////////////////////////////////////////////數字分割


	cv::Mat dig = cv::imread("onlydigitImage3.jpg", CV_LOAD_IMAGE_GRAYSCALE);



	threshold(dig, dig, 150, 255, THRESH_BINARY | THRESH_OTSU);


	cout << "使用 全域OTSU 影像 辨識數字" << endl;


	int again_seg = 0;
	int xor = 0;
	int four_seg = 0;


	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;



	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(dig, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}

	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, row_record[0], neww, (dig.rows) - row_record[0]));


			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				knn(projection_d1);
				knn(projection_d2);
				knn(projection_d3);
				knn(projection_d4);
				knn(projection_d5);
				knn(projection_d6);
				knn(projection_d7);
				knn(projection_d8);
				knn(projection_d9);
				knn(projection_d10);
				knn(projection_d11);
				knn(projection_d12);
				knn(projection_d13);


				cout << endl;
				cout << "數字辨識為: ";



				int numberresult[13];

				vector<string>  ::iterator iter = digitresult[0].begin();
				for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

					cout << *iter;

					int i = 0;
					convertFromString(i, *iter);     ////// string to int

					numberresult[ix] = i;



				}

				cout << endl;
				cout << endl;
				cout << "轉int後數字為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{

					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					xor = 1;
					four_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);


				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = dig(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = dig(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = dig(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = dig(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = dig(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = dig(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = dig(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = dig(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = dig(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = dig(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = dig(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = dig(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = dig(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));

		knn(d1);
		knn(d2);
		knn(d3);
		knn(d4);
		knn(d5);
		knn(d6);
		knn(d7);
		knn(d8);
		knn(d9);
		knn(d10);
		knn(d11);
		knn(d12);
		knn(d13);

		cout << endl;
		cout << "數字辨識為: ";

		stringstream ss;

		int numberresult[13];

		vector<string>  ::iterator iter = digitresult[0].begin();
		for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

			cout << *iter;

			int i = 0;
			convertFromString(i, *iter);     ////// string to int

			numberresult[ix] = i;

			/*ss.str("");
			ss << *iter;
			ss >> numberresult[ix];*/

		}

		cout << endl;
		cout << endl;
		cout << "轉int後數字為: ";

		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "全域OTSU 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;

			xor = 1;

			four_seg = 1;
			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, r_record, neww, (dig.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				knn(projection_d1);
				knn(projection_d2);
				knn(projection_d3);
				knn(projection_d4);
				knn(projection_d5);
				knn(projection_d6);
				knn(projection_d7);
				knn(projection_d8);
				knn(projection_d9);
				knn(projection_d10);
				knn(projection_d11);
				knn(projection_d12);
				knn(projection_d13);


				cout << endl;
				cout << "數字辨識為: ";



				int numberresult[13];

				vector<string>  ::iterator iter = digitresult[0].begin();
				for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

					cout << *iter;

					int i = 0;
					convertFromString(i, *iter);     ////// string to int

					numberresult[ix] = i;



				}

				cout << endl;
				cout << endl;
				cout << "轉int後數字為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					xor = 1;
					four_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}




	}


	if (four_seg == 0)
	{
		four_digitsort(img, bernsen_Location, OTSU__Location, linear_Location, img_num);
	}

}


void again_digitsort(Mat & img, Mat & bernsen_Location, Mat& OTSU__Location, Mat& linear_Location, int& img_num)
{

	Mat new_seg_barcod = bernsen_Location;

	seg_barcode(new_seg_barcod);

	/////////////////////////////////////////////////////////////////////////////////////////數字分割


	cv::Mat dig = cv::imread("onlydigitImage3.jpg", CV_LOAD_IMAGE_GRAYSCALE);



	threshold(dig, dig, 150, 255, THRESH_BINARY | THRESH_OTSU);


	cout << "使用 bernsen 影像 辨識數字" << endl;


	int again_seg = 0;
	int three_seg = 0;

	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;



	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(dig, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}

	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, row_record[0], neww, (dig.rows) - row_record[0]));


			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				knn(projection_d1);
				knn(projection_d2);
				knn(projection_d3);
				knn(projection_d4);
				knn(projection_d5);
				knn(projection_d6);
				knn(projection_d7);
				knn(projection_d8);
				knn(projection_d9);
				knn(projection_d10);
				knn(projection_d11);
				knn(projection_d12);
				knn(projection_d13);


				cout << endl;
				cout << "數字辨識為: ";



				int numberresult[13];

				vector<string>  ::iterator iter = digitresult[0].begin();
				for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

					cout << *iter;

					int i = 0;
					convertFromString(i, *iter);     ////// string to int

					numberresult[ix] = i;



				}

				cout << endl;
				cout << endl;
				cout << "轉int後數字為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域bernsen 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					three_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = dig(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = dig(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = dig(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = dig(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = dig(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = dig(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = dig(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = dig(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = dig(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = dig(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = dig(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = dig(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = dig(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));

		knn(d1);
		knn(d2);
		knn(d3);
		knn(d4);
		knn(d5);
		knn(d6);
		knn(d7);
		knn(d8);
		knn(d9);
		knn(d10);
		knn(d11);
		knn(d12);
		knn(d13);

		cout << endl;
		cout << "數字辨識為: ";

		stringstream ss;

		int numberresult[13];

		vector<string>  ::iterator iter = digitresult[0].begin();
		for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

			cout << *iter;

			int i = 0;
			convertFromString(i, *iter);     ////// string to int

			numberresult[ix] = i;

			/*ss.str("");
			ss << *iter;
			ss >> numberresult[ix];*/

		}

		cout << endl;
		cout << endl;
		cout << "轉int後數字為: ";

		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "全域bernsen 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;
			three_seg = 1;

			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, r_record, neww, (dig.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				knn(projection_d1);
				knn(projection_d2);
				knn(projection_d3);
				knn(projection_d4);
				knn(projection_d5);
				knn(projection_d6);
				knn(projection_d7);
				knn(projection_d8);
				knn(projection_d9);
				knn(projection_d10);
				knn(projection_d11);
				knn(projection_d12);
				knn(projection_d13);


				cout << endl;
				cout << "數字辨識為: ";



				int numberresult[13];

				vector<string>  ::iterator iter = digitresult[0].begin();
				for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

					cout << *iter;

					int i = 0;
					convertFromString(i, *iter);     ////// string to int

					numberresult[ix] = i;



				}

				cout << endl;
				cout << endl;
				cout << "轉int後數字為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域bernsen 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					three_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}




	}

	if (three_seg == 0)
	{
		three_digitsort(img, bernsen_Location, OTSU__Location, linear_Location, img_num);
	}


}


void digitsort(Mat & img, Mat & bernsen_Location, Mat& OTSU__Location, Mat& linear_Location, int& img_num)
{
	int again_seg = 0;
	int bernsen_again_seg = 0;

	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;



	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(img, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}

	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(img.rows, img.cols, CV_8U);

		int neww = (img.cols);
		int newh = (img.rows);



		for (int r = 0; r < img.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < img.cols; c++)
			{
				if (img.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = img(Rect(0, row_record[0], neww, (img.rows) - row_record[0]));


			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				knn(projection_d1);
				knn(projection_d2);
				knn(projection_d3);
				knn(projection_d4);
				knn(projection_d5);
				knn(projection_d6);
				knn(projection_d7);
				knn(projection_d8);
				knn(projection_d9);
				knn(projection_d10);
				knn(projection_d11);
				knn(projection_d12);
				knn(projection_d13);


				cout << endl;
				cout << "數字辨識為: ";



				int numberresult[13];

				vector<string>  ::iterator iter = digitresult[0].begin();
				for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

					cout << *iter;

					int i = 0;
					convertFromString(i, *iter);     ////// string to int

					numberresult[ix] = i;



				}

				cout << endl;
				cout << endl;
				cout << "轉int後數字為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "區域OTSU 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					bernsen_again_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}

		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = img(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = img(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = img(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = img(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = img(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = img(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = img(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = img(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = img(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = img(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = img(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = img(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = img(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));

		knn(d1);
		knn(d2);
		knn(d3);
		knn(d4);
		knn(d5);
		knn(d6);
		knn(d7);
		knn(d8);
		knn(d9);
		knn(d10);
		knn(d11);
		knn(d12);
		knn(d13);

		cout << endl;
		cout << "數字辨識為: ";

		stringstream ss;

		int numberresult[13];

		vector<string>  ::iterator iter = digitresult[0].begin();
		for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

			cout << *iter;

			int i = 0;
			convertFromString(i, *iter);     ////// string to int

			numberresult[ix] = i;

			/*ss.str("");
			ss << *iter;
			ss >> numberresult[ix];*/

		}

		cout << endl;
		cout << endl;
		cout << "轉int後數字為: ";

		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "區域OTSU 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;
			bernsen_again_seg = 1;

			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(img.rows, img.cols, CV_8U);

		int neww = (img.cols);
		int newh = (img.rows);



		for (int r = 0; r < img.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < img.cols; c++)
			{
				if (img.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = img(Rect(0, r_record, neww, (img.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				knn(projection_d1);
				knn(projection_d2);
				knn(projection_d3);
				knn(projection_d4);
				knn(projection_d5);
				knn(projection_d6);
				knn(projection_d7);
				knn(projection_d8);
				knn(projection_d9);
				knn(projection_d10);
				knn(projection_d11);
				knn(projection_d12);
				knn(projection_d13);


				cout << endl;
				cout << "數字辨識為: ";



				int numberresult[13];

				vector<string>  ::iterator iter = digitresult[0].begin();
				for (int ix = 0; iter != digitresult[0].end(); ++iter, ++ix) {

					cout << *iter;

					int i = 0;
					convertFromString(i, *iter);     ////// string to int

					numberresult[ix] = i;



				}

				cout << endl;
				cout << endl;
				cout << "轉int後數字為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "區域OTSU 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					bernsen_again_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);

				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}

		}




	}

	if (bernsen_again_seg == 0)
	{
		again_digitsort(img, bernsen_Location, OTSU__Location, linear_Location, img_num);

	}

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

void xor_four_digitsort(Mat & img, Mat & bernsen_Location, Mat& OTSU__Location, Mat& linear_Location, int& img_num)
{

	Mat new_seg_barcod = linear_Location;

	seg_barcode2(new_seg_barcod);

	/////////////////////////////////////////////////////////////////////////////////////////數字分割


	cv::Mat dig = cv::imread("onlydigitImage3.jpg", CV_LOAD_IMAGE_GRAYSCALE);



	threshold(dig, dig, 150, 255, THRESH_BINARY | THRESH_OTSU);


	cout << "使用 全域OTSU 影像 辨識數字" << endl;


	int again_seg = 0;

	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;

	int svm_seg = 0;

	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(dig, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}

	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, row_record[0], neww, (dig.rows) - row_record[0]));


			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				cout << endl;
				int numberresult[13];

				deal(projection_d1, digit_xor);
				numberresult[0] = digit_xor;
				deal(projection_d2, digit_xor);
				numberresult[1] = digit_xor;
				deal(projection_d3, digit_xor);
				numberresult[2] = digit_xor;
				deal(projection_d4, digit_xor);
				numberresult[3] = digit_xor;
				deal(projection_d5, digit_xor);
				numberresult[4] = digit_xor;
				deal(projection_d6, digit_xor);
				numberresult[5] = digit_xor;
				deal(projection_d7, digit_xor);
				numberresult[6] = digit_xor;
				deal(projection_d8, digit_xor);
				numberresult[7] = digit_xor;
				deal(projection_d9, digit_xor);
				numberresult[8] = digit_xor;
				deal(projection_d10, digit_xor);
				numberresult[9] = digit_xor;
				deal(projection_d11, digit_xor);
				numberresult[10] = digit_xor;
				deal(projection_d12, digit_xor);
				numberresult[11] = digit_xor;
				deal(projection_d13, digit_xor);
				numberresult[12] = digit_xor;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";





				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{

					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					svm_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);


				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = dig(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = dig(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = dig(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = dig(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = dig(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = dig(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = dig(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = dig(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = dig(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = dig(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = dig(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = dig(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = dig(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));


		cout << endl;
		int numberresult[13];

		deal(d1, digit_xor);
		numberresult[0] = digit_xor;
		deal(d2, digit_xor);
		numberresult[1] = digit_xor;
		deal(d3, digit_xor);
		numberresult[2] = digit_xor;
		deal(d4, digit_xor);
		numberresult[3] = digit_xor;
		deal(d5, digit_xor);
		numberresult[4] = digit_xor;
		deal(d6, digit_xor);
		numberresult[5] = digit_xor;
		deal(d7, digit_xor);
		numberresult[6] = digit_xor;
		deal(d8, digit_xor);
		numberresult[7] = digit_xor;
		deal(d9, digit_xor);
		numberresult[8] = digit_xor;
		deal(d10, digit_xor);
		numberresult[9] = digit_xor;
		deal(d11, digit_xor);
		numberresult[10] = digit_xor;
		deal(d12, digit_xor);
		numberresult[11] = digit_xor;
		deal(d13, digit_xor);
		numberresult[12] = digit_xor;





		cout << endl;
		cout << endl;

		cout << "數字辨識為: ";





		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "全域OTSU 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;

			svm_seg = 1;

			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, r_record, neww, (dig.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));



				cout << endl;
				int numberresult[13];

				deal(projection_d1, digit_xor);
				numberresult[0] = digit_xor;
				deal(projection_d2, digit_xor);
				numberresult[1] = digit_xor;
				deal(projection_d3, digit_xor);
				numberresult[2] = digit_xor;
				deal(projection_d4, digit_xor);
				numberresult[3] = digit_xor;
				deal(projection_d5, digit_xor);
				numberresult[4] = digit_xor;
				deal(projection_d6, digit_xor);
				numberresult[5] = digit_xor;
				deal(projection_d7, digit_xor);
				numberresult[6] = digit_xor;
				deal(projection_d8, digit_xor);
				numberresult[7] = digit_xor;
				deal(projection_d9, digit_xor);
				numberresult[8] = digit_xor;
				deal(projection_d10, digit_xor);
				numberresult[9] = digit_xor;
				deal(projection_d11, digit_xor);
				numberresult[10] = digit_xor;
				deal(projection_d12, digit_xor);
				numberresult[11] = digit_xor;
				deal(projection_d13, digit_xor);
				numberresult[12] = digit_xor;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";





				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					svm_seg = 1;
					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}




	}

	if (svm_seg == 0)
	{
		svm_digitsort(img, bernsen_Location, OTSU__Location, linear_Location, img_num);
	}



}

void xor_three_digitsort(Mat & img, Mat & bernsen_Location, Mat& OTSU__Location, Mat& linear_Location, int& img_num)
{

	Mat new_seg_barcod = OTSU__Location;

	seg_barcode(new_seg_barcod);

	/////////////////////////////////////////////////////////////////////////////////////////數字分割


	cv::Mat dig = cv::imread("onlydigitImage3.jpg", CV_LOAD_IMAGE_GRAYSCALE);



	threshold(dig, dig, 150, 255, THRESH_BINARY | THRESH_OTSU);


	cout << "使用 全域OTSU 影像 辨識數字" << endl;


	int again_seg = 0;
	int four_seg = 0;


	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;



	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(dig, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}

	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, row_record[0], neww, (dig.rows) - row_record[0]));


			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				cout << endl;
				int numberresult[13];

				deal(projection_d1, digit_xor);
				numberresult[0] = digit_xor;
				deal(projection_d2, digit_xor);
				numberresult[1] = digit_xor;
				deal(projection_d3, digit_xor);
				numberresult[2] = digit_xor;
				deal(projection_d4, digit_xor);
				numberresult[3] = digit_xor;
				deal(projection_d5, digit_xor);
				numberresult[4] = digit_xor;
				deal(projection_d6, digit_xor);
				numberresult[5] = digit_xor;
				deal(projection_d7, digit_xor);
				numberresult[6] = digit_xor;
				deal(projection_d8, digit_xor);
				numberresult[7] = digit_xor;
				deal(projection_d9, digit_xor);
				numberresult[8] = digit_xor;
				deal(projection_d10, digit_xor);
				numberresult[9] = digit_xor;
				deal(projection_d11, digit_xor);
				numberresult[10] = digit_xor;
				deal(projection_d12, digit_xor);
				numberresult[11] = digit_xor;
				deal(projection_d13, digit_xor);
				numberresult[12] = digit_xor;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";





				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{

					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					four_seg = 1;
					bar_or_digits = 2;

					check(numberresult, img_num);


				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = dig(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = dig(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = dig(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = dig(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = dig(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = dig(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = dig(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = dig(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = dig(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = dig(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = dig(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = dig(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = dig(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));


		cout << endl;
		int numberresult[13];

		deal(d1, digit_xor);
		numberresult[0] = digit_xor;
		deal(d2, digit_xor);
		numberresult[1] = digit_xor;
		deal(d3, digit_xor);
		numberresult[2] = digit_xor;
		deal(d4, digit_xor);
		numberresult[3] = digit_xor;
		deal(d5, digit_xor);
		numberresult[4] = digit_xor;
		deal(d6, digit_xor);
		numberresult[5] = digit_xor;
		deal(d7, digit_xor);
		numberresult[6] = digit_xor;
		deal(d8, digit_xor);
		numberresult[7] = digit_xor;
		deal(d9, digit_xor);
		numberresult[8] = digit_xor;
		deal(d10, digit_xor);
		numberresult[9] = digit_xor;
		deal(d11, digit_xor);
		numberresult[10] = digit_xor;
		deal(d12, digit_xor);
		numberresult[11] = digit_xor;
		deal(d13, digit_xor);
		numberresult[12] = digit_xor;





		cout << endl;
		cout << endl;

		cout << "數字辨識為: ";





		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "全域OTSU 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;

			four_seg = 1;
			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, r_record, neww, (dig.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));



				cout << endl;
				int numberresult[13];

				deal(projection_d1, digit_xor);
				numberresult[0] = digit_xor;
				deal(projection_d2, digit_xor);
				numberresult[1] = digit_xor;
				deal(projection_d3, digit_xor);
				numberresult[2] = digit_xor;
				deal(projection_d4, digit_xor);
				numberresult[3] = digit_xor;
				deal(projection_d5, digit_xor);
				numberresult[4] = digit_xor;
				deal(projection_d6, digit_xor);
				numberresult[5] = digit_xor;
				deal(projection_d7, digit_xor);
				numberresult[6] = digit_xor;
				deal(projection_d8, digit_xor);
				numberresult[7] = digit_xor;
				deal(projection_d9, digit_xor);
				numberresult[8] = digit_xor;
				deal(projection_d10, digit_xor);
				numberresult[9] = digit_xor;
				deal(projection_d11, digit_xor);
				numberresult[10] = digit_xor;
				deal(projection_d12, digit_xor);
				numberresult[11] = digit_xor;
				deal(projection_d13, digit_xor);
				numberresult[12] = digit_xor;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";





				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域OTSU 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;

					four_seg = 1;
					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}




	}


	if (four_seg == 0)
	{
		xor_four_digitsort(img, bernsen_Location, OTSU__Location, linear_Location, img_num);
	}


}


void xor_again_digitsort(Mat & img, Mat & bernsen_Location, Mat& OTSU__Location, Mat& linear_Location, int& img_num)
{

	Mat new_seg_barcod = bernsen_Location;

	seg_barcode(new_seg_barcod);

	/////////////////////////////////////////////////////////////////////////////////////////數字分割


	cv::Mat dig = cv::imread("onlydigitImage3.jpg", CV_LOAD_IMAGE_GRAYSCALE);



	threshold(dig, dig, 150, 255, THRESH_BINARY | THRESH_OTSU);


	cout << "使用 bernsen 影像 辨識數字" << endl;


	int again_seg = 0;
	int three_seg = 0;

	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;



	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(dig, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}

	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, row_record[0], neww, (dig.rows) - row_record[0]));


			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				cout << endl;
				int numberresult[13];

				deal(projection_d1, digit_xor);
				numberresult[0] = digit_xor;
				deal(projection_d2, digit_xor);
				numberresult[1] = digit_xor;
				deal(projection_d3, digit_xor);
				numberresult[2] = digit_xor;
				deal(projection_d4, digit_xor);
				numberresult[3] = digit_xor;
				deal(projection_d5, digit_xor);
				numberresult[4] = digit_xor;
				deal(projection_d6, digit_xor);
				numberresult[5] = digit_xor;
				deal(projection_d7, digit_xor);
				numberresult[6] = digit_xor;
				deal(projection_d8, digit_xor);
				numberresult[7] = digit_xor;
				deal(projection_d9, digit_xor);
				numberresult[8] = digit_xor;
				deal(projection_d10, digit_xor);
				numberresult[9] = digit_xor;
				deal(projection_d11, digit_xor);
				numberresult[10] = digit_xor;
				deal(projection_d12, digit_xor);
				numberresult[11] = digit_xor;
				deal(projection_d13, digit_xor);
				numberresult[12] = digit_xor;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";



				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域bernsen 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					three_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = dig(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = dig(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = dig(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = dig(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = dig(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = dig(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = dig(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = dig(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = dig(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = dig(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = dig(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = dig(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = dig(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));


		cout << endl;
		int numberresult[13];

		deal(d1, digit_xor);
		numberresult[0] = digit_xor;
		deal(d2, digit_xor);
		numberresult[1] = digit_xor;
		deal(d3, digit_xor);
		numberresult[2] = digit_xor;
		deal(d4, digit_xor);
		numberresult[3] = digit_xor;
		deal(d5, digit_xor);
		numberresult[4] = digit_xor;
		deal(d6, digit_xor);
		numberresult[5] = digit_xor;
		deal(d7, digit_xor);
		numberresult[6] = digit_xor;
		deal(d8, digit_xor);
		numberresult[7] = digit_xor;
		deal(d9, digit_xor);
		numberresult[8] = digit_xor;
		deal(d10, digit_xor);
		numberresult[9] = digit_xor;
		deal(d11, digit_xor);
		numberresult[10] = digit_xor;
		deal(d12, digit_xor);
		numberresult[11] = digit_xor;
		deal(d13, digit_xor);
		numberresult[12] = digit_xor;





		cout << endl;
		cout << endl;

		cout << "數字辨識為: ";





		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "全域bernsen 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;
			three_seg = 1;

			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(dig.rows, dig.cols, CV_8U);

		int neww = (dig.cols);
		int newh = (dig.rows);



		for (int r = 0; r < dig.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < dig.cols; c++)
			{
				if (dig.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = dig(Rect(0, r_record, neww, (dig.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));





				cout << endl;
				int numberresult[13];

				deal(projection_d1, digit_xor);
				numberresult[0] = digit_xor;
				deal(projection_d2, digit_xor);
				numberresult[1] = digit_xor;
				deal(projection_d3, digit_xor);
				numberresult[2] = digit_xor;
				deal(projection_d4, digit_xor);
				numberresult[3] = digit_xor;
				deal(projection_d5, digit_xor);
				numberresult[4] = digit_xor;
				deal(projection_d6, digit_xor);
				numberresult[5] = digit_xor;
				deal(projection_d7, digit_xor);
				numberresult[6] = digit_xor;
				deal(projection_d8, digit_xor);
				numberresult[7] = digit_xor;
				deal(projection_d9, digit_xor);
				numberresult[8] = digit_xor;
				deal(projection_d10, digit_xor);
				numberresult[9] = digit_xor;
				deal(projection_d11, digit_xor);
				numberresult[10] = digit_xor;
				deal(projection_d12, digit_xor);
				numberresult[11] = digit_xor;
				deal(projection_d13, digit_xor);
				numberresult[12] = digit_xor;





				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";



				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "全域bernsen 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					three_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}
		}




	}

	if (three_seg == 0)
	{
		xor_three_digitsort(img, bernsen_Location, OTSU__Location, linear_Location, img_num);
	}


}

void xor_digitsort(Mat & img, Mat & bernsen_Location, Mat& OTSU__Location, Mat& linear_Location, int& img_num)
{
	int again_seg = 0;
	int bernsen_again_seg = 0;

	cv::Mat labelImage5;
	cv::Mat stats5, centroids5;



	vector< myclass > vect;

	int nLabels5 = cv::connectedComponentsWithStats(img, labelImage5, stats5, centroids5, 8, CV_32S);/////八連通


																									 //cout << "nLabels5:  " << nLabels5 << endl;



	std::vector<cv::Vec3b> colorsd(nLabels5);
	colorsd[0] = cv::Vec3b(0, 0, 0);

	for (int label = 1; label < nLabels5; ++label)
	{
		//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
		//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
		//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;




	}

	if (nLabels5 != 14)
	{
		cout << endl;
		cout << "重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(img.rows, img.cols, CV_8U);

		int neww = (img.cols);
		int newh = (img.rows);



		for (int r = 0; r < img.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < img.cols; c++)
			{
				if (img.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;


		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (rowHiss[i] == 0 && rowHiss[i + 1] != 0)                  ////////////////////////////////   找第一間格
			{

				row_record.push_back((i + 1));


			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = img(Rect(0, row_record[0], neww, (img.rows) - row_record[0]));



			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));






				cout << endl;
				int numberresult[13];




				deal(projection_d1, digit_xor);
				numberresult[0] = digit_xor;
				deal(projection_d2, digit_xor);
				numberresult[1] = digit_xor;
				deal(projection_d3, digit_xor);
				numberresult[2] = digit_xor;
				deal(projection_d4, digit_xor);
				numberresult[3] = digit_xor;
				deal(projection_d5, digit_xor);
				numberresult[4] = digit_xor;
				deal(projection_d6, digit_xor);
				numberresult[5] = digit_xor;
				deal(projection_d7, digit_xor);
				numberresult[6] = digit_xor;
				deal(projection_d8, digit_xor);
				numberresult[7] = digit_xor;
				deal(projection_d9, digit_xor);
				numberresult[8] = digit_xor;
				deal(projection_d10, digit_xor);
				numberresult[9] = digit_xor;
				deal(projection_d11, digit_xor);
				numberresult[10] = digit_xor;
				deal(projection_d12, digit_xor);
				numberresult[11] = digit_xor;
				deal(projection_d13, digit_xor);
				numberresult[12] = digit_xor;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "區域OTSU 找第一間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					bernsen_again_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);
				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}

		}



	}

	/////////////////////////////////////////////////////////檢測字為13碼

	if (nLabels5 == 14)
	{
		cout << endl;
		cout << "檢測到13碼: " << endl;
		cout << endl;
		for (int label = 1; label < nLabels5; ++label)
		{
			//cout << "CC_STAT_LEFT   = " << stats5.at<int>(label, cv::CC_STAT_LEFT) << endl;
			//cout << "CC_STAT_TOP   = " << stats5.at<int>(label, cv::CC_STAT_TOP) << endl;
			//cout << "CC_STAT_WIDTH   = " << stats5.at<int>(label, cv::CC_STAT_WIDTH) << endl;



			myclass my(stats5.at<int>(label, cv::CC_STAT_LEFT), stats5.at<int>(label, cv::CC_STAT_TOP), stats5.at<int>(label, cv::CC_STAT_WIDTH), stats5.at<int>(label, cv::CC_STAT_HEIGHT));
			vect.push_back(my);



		}


		sort(vect.begin(), vect.end());//排序位置



									   /*for (int i = 0; i < vect.size(); i++)
									   {
									   cout << "(" << vect[i].first << "," << vect[i].second << "," << vect[i].third << "," << vect[i].fourth << ")\n";
									   }*/



		Mat d1 = img(Rect(vect[0].first, vect[0].second, vect[0].third, vect[0].fourth));
		Mat d2 = img(Rect(vect[1].first, vect[1].second, vect[1].third, vect[1].fourth));
		Mat d3 = img(Rect(vect[2].first, vect[2].second, vect[2].third, vect[2].fourth));
		Mat d4 = img(Rect(vect[3].first, vect[3].second, vect[3].third, vect[3].fourth));
		Mat d5 = img(Rect(vect[4].first, vect[4].second, vect[4].third, vect[4].fourth));
		Mat d6 = img(Rect(vect[5].first, vect[5].second, vect[5].third, vect[5].fourth));
		Mat d7 = img(Rect(vect[6].first, vect[6].second, vect[6].third, vect[6].fourth));
		Mat d8 = img(Rect(vect[7].first, vect[7].second, vect[7].third, vect[7].fourth));
		Mat d9 = img(Rect(vect[8].first, vect[8].second, vect[8].third, vect[8].fourth));
		Mat d10 = img(Rect(vect[9].first, vect[9].second, vect[9].third, vect[9].fourth));
		Mat d11 = img(Rect(vect[10].first, vect[10].second, vect[10].third, vect[10].fourth));
		Mat d12 = img(Rect(vect[11].first, vect[11].second, vect[11].third, vect[11].fourth));
		Mat d13 = img(Rect(vect[12].first, vect[12].second, vect[12].third, vect[12].fourth));

		cout << endl;
		int numberresult[13];

		deal(d1, digit_xor);
		numberresult[0] = digit_xor;
		deal(d2, digit_xor);
		numberresult[1] = digit_xor;
		deal(d3, digit_xor);
		numberresult[2] = digit_xor;
		deal(d4, digit_xor);
		numberresult[3] = digit_xor;
		deal(d5, digit_xor);
		numberresult[4] = digit_xor;
		deal(d6, digit_xor);
		numberresult[5] = digit_xor;
		deal(d7, digit_xor);
		numberresult[6] = digit_xor;
		deal(d8, digit_xor);
		numberresult[7] = digit_xor;
		deal(d9, digit_xor);
		numberresult[8] = digit_xor;
		deal(d10, digit_xor);
		numberresult[9] = digit_xor;
		deal(d11, digit_xor);
		numberresult[10] = digit_xor;
		deal(d12, digit_xor);
		numberresult[11] = digit_xor;
		deal(d13, digit_xor);
		numberresult[12] = digit_xor;




		cout << endl;
		cout << endl;

		cout << "數字辨識為: ";

		for (int i = 0; i < 13; i++)
		{
			cout << numberresult[i];
		}
		cout << endl;
		cout << endl;

		//////////////////////////////////////////////////////////////////// 檢查碼驗證



		int g1, g2, g3, g4;
		g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

		g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

		g3 = (g1 + g2) % 10;


		g4 = (10 - g3);

		if (g4 == 10)
		{
			g4 = 0;
		}

		cout << endl;

		cout << "check digits: " << g4 << " " << endl;

		if (g4 == numberresult[12])
		{
			cout << "數字檢查碼為正確" << endl;
			cout << "區域OTSU 連通切割數字" << endl;
			digits2_success += 1;
			again_seg = 1;
			onltsuc2 = 1;
			bernsen_again_seg = 1;

			bar_or_digits = 2;

			check(numberresult, img_num);
		}
		else
		{
			cout << "數字檢查碼不正確" << endl;

		}










		digits_success = 1;

		digitresult[0].clear();


		imwrite("d1.jpg", d1);
		imwrite("d2.jpg", d2);
		imwrite("d3.jpg", d3);
		imwrite("d4.jpg", d4);
		imwrite("d5.jpg", d5);
		imwrite("d6.jpg", d6);
		imwrite("d7.jpg", d7);
		imwrite("d8.jpg", d8);
		imwrite("d9.jpg", d9);
		imwrite("d10.jpg", d10);
		imwrite("d11.jpg", d11);
		imwrite("d12.jpg", d12);
		imwrite("d13.jpg", d13);
	}


	if (again_seg == 0)                     ///////////////////////////////////////////////////   第二次重新切割
	{

		cout << endl;
		cout << "第二次重新切割數字 " << endl;
		cout << endl;

		Mat Horizontalprojection_dig;

		Horizontalprojection_dig.create(img.rows, img.cols, CV_8U);

		int neww = (img.cols);
		int newh = (img.rows);



		for (int r = 0; r < img.rows; r++)
		{
			int a = 0;
			for (int c = 0; c < img.cols; c++)
			{
				if (img.at<uchar>(r, c) == 0) Horizontalprojection_dig.at<uchar>(r, a++) = 0;


			}
		}


		std::vector<int> rowHiss(Horizontalprojection_dig.rows);

		for (int i = 0; i < Horizontalprojection_dig.rows; i++)
		{
			for (int j = 0; j < Horizontalprojection_dig.cols; j++)
			{
				if (Horizontalprojection_dig.at<uchar>(i, j) != 0)
				{
					rowHiss[i]++;
				}
			}


		}


		vector<int> row_record;

		int max_row_record = 0, r_record = 0;

		for (size_t i = 0; i < rowHiss.size() - 1; i++)
		{
			//cout << rowHiss[i] << " " << endl;
			if (max_row_record < rowHiss[i])             /////////////////////////////////////      找最大間格   
			{

				max_row_record = rowHiss[i];


				row_record.push_back((i));
				r_record = i;

			}




		}

		if (row_record.empty())
		{
			cout << "row_record is empty" << endl;

		}
		else
		{

			Mat row_record_img = img(Rect(0, r_record, neww, (img.rows) - r_record));
			//cout << "r_record: " << r_record << endl;

			//imshow("row_record_img.", row_record_img);

			//imshow("Horizontalprojection_dig.", Horizontalprojection_dig);


			cv::Mat row_record_labelImage;
			cv::Mat row_record_stats, row_record_centroids;

			int row_record_Labels = cv::connectedComponentsWithStats(row_record_img, row_record_labelImage, row_record_stats, row_record_centroids, 8, CV_32S);/////八連通







																																							   ////////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
			Mat Verticalprojection;
			Verticalprojection.create(row_record_img.rows, row_record_img.cols, CV_8U);


			for (int c = 0; c < row_record_img.cols; c++)
			{
				int a = 0;
				for (int r = 0; r < row_record_img.rows; r++)
				{

					if (row_record_img.at<uchar>(r, c) == 0) Verticalprojection.at<uchar>(a++, c) = 0;
				}
			}

			//imshow("Verticalprojection", Verticalprojection);
			imwrite("Verticalprojection.jpg", Verticalprojection);



			std::vector<int> colHiss(Verticalprojection.cols);

			for (int i = 0; i < Verticalprojection.cols; i++)
			{
				for (int j = 0; j < Verticalprojection.rows; j++)
				{
					if (Verticalprojection.at<uchar>(j, i) != 0)
					{
						colHiss[i]++;
					}
				}


			}


			vector<int> digitsrecord;          ////////////////////////////////////////////////////////////  投影切割

			int ind = 0;

			for (size_t i = 0; i < colHiss.size() - 1; i++)
			{
				//cout << colHiss[i]<<" " ;

				if (colHiss[i] < Verticalprojection.rows)
				{
					if (colHiss[i] == 0 && colHiss[i + 1] != 0 || colHiss[i] != 0 && colHiss[i + 1] == 0)
					{

						digitsrecord.push_back((i + 1));
						ind += 1;

					}
				}
			}
			cout << endl;

			if (ind == 26)
			{


				Mat projection_d1 = row_record_img(Rect(digitsrecord[0], 1, (digitsrecord[1] - digitsrecord[0]), row_record_img.rows - 1));
				Mat projection_d2 = row_record_img(Rect(digitsrecord[2], 1, (digitsrecord[3] - digitsrecord[2]), row_record_img.rows - 1));
				Mat projection_d3 = row_record_img(Rect(digitsrecord[4], 1, (digitsrecord[5] - digitsrecord[4]), row_record_img.rows - 1));
				Mat projection_d4 = row_record_img(Rect(digitsrecord[6], 1, (digitsrecord[7] - digitsrecord[6]), row_record_img.rows - 1));
				Mat projection_d5 = row_record_img(Rect(digitsrecord[8], 1, (digitsrecord[9] - digitsrecord[8]), row_record_img.rows - 1));
				Mat projection_d6 = row_record_img(Rect(digitsrecord[10], 1, (digitsrecord[11] - digitsrecord[10]), row_record_img.rows - 1));
				Mat projection_d7 = row_record_img(Rect(digitsrecord[12], 1, (digitsrecord[13] - digitsrecord[12]), row_record_img.rows - 1));
				Mat projection_d8 = row_record_img(Rect(digitsrecord[14], 1, (digitsrecord[15] - digitsrecord[14]), row_record_img.rows - 1));
				Mat projection_d9 = row_record_img(Rect(digitsrecord[16], 1, (digitsrecord[17] - digitsrecord[16]), row_record_img.rows - 1));
				Mat projection_d10 = row_record_img(Rect(digitsrecord[18], 1, (digitsrecord[19] - digitsrecord[18]), row_record_img.rows - 1));
				Mat projection_d11 = row_record_img(Rect(digitsrecord[20], 1, (digitsrecord[21] - digitsrecord[20]), row_record_img.rows - 1));
				Mat projection_d12 = row_record_img(Rect(digitsrecord[22], 1, (digitsrecord[23] - digitsrecord[22]), row_record_img.rows - 1));
				Mat projection_d13 = row_record_img(Rect(digitsrecord[24], 1, (digitsrecord[25] - digitsrecord[24]), row_record_img.rows - 1));




				cout << endl;
				int numberresult[13];

				deal(projection_d1, digit_xor);
				numberresult[0] = digit_xor;
				deal(projection_d2, digit_xor);
				numberresult[1] = digit_xor;
				deal(projection_d3, digit_xor);
				numberresult[2] = digit_xor;
				deal(projection_d4, digit_xor);
				numberresult[3] = digit_xor;
				deal(projection_d5, digit_xor);
				numberresult[4] = digit_xor;
				deal(projection_d6, digit_xor);
				numberresult[5] = digit_xor;
				deal(projection_d7, digit_xor);
				numberresult[6] = digit_xor;
				deal(projection_d8, digit_xor);
				numberresult[7] = digit_xor;
				deal(projection_d9, digit_xor);
				numberresult[8] = digit_xor;
				deal(projection_d10, digit_xor);
				numberresult[9] = digit_xor;
				deal(projection_d11, digit_xor);
				numberresult[10] = digit_xor;
				deal(projection_d12, digit_xor);
				numberresult[11] = digit_xor;
				deal(projection_d13, digit_xor);
				numberresult[12] = digit_xor;




				cout << endl;
				cout << endl;

				cout << "數字辨識為: ";

				for (int i = 0; i < 13; i++)
				{
					cout << numberresult[i];
				}
				cout << endl;
				cout << endl;





				int g1, g2, g3, g4;
				g1 = (numberresult[11] + numberresult[9] + numberresult[7] + numberresult[5] + numberresult[3] + numberresult[1]) * 3;

				g2 = (numberresult[10] + numberresult[8] + numberresult[6] + numberresult[4] + numberresult[2] + numberresult[0]);

				g3 = (g1 + g2) % 10;


				g4 = (10 - g3);

				if (g4 == 10)
				{
					g4 = 0;
				}

				cout << endl;

				cout << "check digits: " << g4 << " " << endl;

				if (g4 == numberresult[12])
				{
					cout << "數字檢查碼為正確" << endl;
					cout << "區域OTSU 找最大間格切割數字" << endl;
					digits2_success += 1;
					again_seg = 1;
					onltsuc2 = 1;
					bernsen_again_seg = 1;

					bar_or_digits = 2;

					check(numberresult, img_num);

				}
				else
				{
					cout << "數字檢查碼不正確" << endl;

				}


				cout << endl;







				digits_success = 1;

				digitresult[0].clear();


				imwrite("p1.jpg", projection_d1);
				imwrite("p2.jpg", projection_d2);
				imwrite("p3.jpg", projection_d3);
				imwrite("p4.jpg", projection_d4);
				imwrite("p5.jpg", projection_d5);
				imwrite("p6.jpg", projection_d6);
				imwrite("p7.jpg", projection_d7);
				imwrite("p8.jpg", projection_d8);
				imwrite("p9.jpg", projection_d9);
				imwrite("p10.jpg", projection_d10);
				imwrite("p11.jpg", projection_d11);
				imwrite("p12.jpg", projection_d12);
				imwrite("p13.jpg", projection_d13);

			}

		}




	}

	if (bernsen_again_seg == 0)
	{
		xor_again_digitsort(img, bernsen_Location, OTSU__Location, linear_Location, img_num);

	}

}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////
void Rotation_contour(Mat& barcode, Mat& nbm, float& blob_angle)
{
	std::vector<cv::Vec4i> lines;
	cv::Size bsize = barcode.size();
	cv::HoughLinesP(barcode, lines, 1, CV_PI / 180, 100, bsize.width / 2.f, 20);


	cv::Mat disp_lines(bsize, CV_8UC1, cv::Scalar(0, 0, 0));
	double angle = 0.;
	unsigned nb_lines = lines.size();
	for (unsigned i = 0; i < nb_lines; ++i)
	{
		cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]),
			cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0));
		angle += atan2((double)lines[i][3] - lines[i][1],
			(double)lines[i][2] - lines[i][0]);
	}
	angle /= nb_lines; // mean angle, in radians.

	float r = angle * 180 / CV_PI;


	//cout << "angle:" << r << endl;



	cv::Point2f center = Point(barcode.cols / 2.0, barcode.rows / 2.0);

	double scale = 1.0;

	/* nbm = getRotationMatrix2D(center, blob_angle, scale);*/

	Mat rot_mat = getRotationMatrix2D(center, blob_angle, scale);
	cv::Rect bbox = cv::RotatedRect(center, barcode.size(), blob_angle).boundingRect();
	rot_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;
	warpAffine(barcode, nbm, rot_mat, bbox.size());

}

////////////////////////////////////////////////////////////////////////////////

void Rotation_barcode(Mat& barcode, Mat& nbm)
{
	std::vector<cv::Vec4i> lines;
	cv::Size bsize = barcode.size();
	cv::HoughLinesP(barcode, lines, 1, CV_PI / 180, 100, bsize.width / 2.f, 20);


	cv::Mat disp_lines(bsize, CV_8UC1, cv::Scalar(0, 0, 0));
	double angle = 0.;
	unsigned nb_lines = lines.size();
	for (unsigned i = 0; i < nb_lines; ++i)
	{
		cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]),
			cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0));
		angle += atan2((double)lines[i][3] - lines[i][1],
			(double)lines[i][2] - lines[i][0]);
	}
	angle /= nb_lines; // mean angle, in radians.

	float r = angle * 180 / CV_PI;

	if (r > 0)
	{
		r -= 90;
	}
	else
	{
		r += 90;
	}
	//cout << "angle:" << r << endl;



	cv::Point2f center = Point(barcode.cols / 2.0, barcode.rows / 2.0);

	double scale = 1.0;

	Mat rot_mat = getRotationMatrix2D(center, r, scale);
	cv::Rect bbox = cv::RotatedRect(center, barcode.size(), r).boundingRect();
	rot_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;
	warpAffine(barcode, nbm, rot_mat, bbox.size());
}

////////////////////////////////////////////////////////////////////////////////

void linearTrans(const Mat &src, Mat &dst) {
	dst.create(src.size(), src.type());
	int widthLimit = src.channels() * src.cols;
	for (int iH = 0; iH < src.rows; iH++) {
		const uchar *curPtr = src.ptr<const uchar>(iH);
		uchar *dstPtr = dst.ptr<uchar>(iH);
		for (int iW = 0; iW < widthLimit; iW++) {
			dstPtr[iW] = saturate_cast<uchar>(1.5*curPtr[iW] + 30);
		}
	}
}

///////////////////////////////////////////////////////////////////////////////////

void  Location_seg2(Mat& image, int& reseg, int& img_num)
{
	Mat grad_x, grad_y, xx;
	Mat abs_grad_x, abs_grad_y;

	Mat co_img, bernsen, all_OTSU, all_adaptive, Hist_bernsen, Gamma, linear;

	cvtColor(image, co_img, CV_RGB2GRAY);


	equalizeHist(co_img, Hist_bernsen);

	cv::Mat sharp_resultl, sharp_OTSU;


	sharpenImage1(co_img, sharp_resultl);


	bernsen = thresh_bernsen(co_img, 25, 40);

	adaptiveThreshold(co_img, all_adaptive, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 65, 0);

	threshold(sharp_resultl, all_OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);


	threshold(co_img, sharp_OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);

	Hist_bernsen = thresh_bernsen(Hist_bernsen, 25, 40);


	linearTrans(co_img, linear);
	linear = thresh_bernsen(linear, 25, 40);


	bitwise_not(bernsen, bernsen);
	bitwise_not(all_OTSU, all_OTSU);
	bitwise_not(all_adaptive, all_adaptive);
	bitwise_not(sharp_OTSU, sharp_OTSU);
	bitwise_not(Hist_bernsen, Hist_bernsen);
	bitwise_not(linear, linear);






	Sobel(co_img, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);  //轉成CV_8U     水平增強



	Sobel(co_img, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);// 垂直增強

	Mat Horizontal, vertical, OTSU, OTSU2, CLOSE;

	subtract(abs_grad_x, abs_grad_y, Horizontal);//X-Y
	subtract(abs_grad_y, abs_grad_x, vertical);//Y-X

	threshold(Horizontal, OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);//OTSU 二值化
	threshold(vertical, OTSU2, 150, 255, THRESH_BINARY | THRESH_OTSU);//OTSU 二值化

																	  ///////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
	Mat Verticalseg;
	Verticalseg.create(OTSU2.rows, OTSU2.cols, CV_8U);


	for (int c = 0; c < OTSU2.cols; c++)
	{
		int a = 0;
		for (int r = 0; r < OTSU2.rows; r++)
		{

			if (OTSU2.at<uchar>(r, c) == 0) Verticalseg.at<uchar>(a++, c) = 0;
		}
	}

	////imshow("Verticalseg", Verticalseg);

	Mat Horizontalseg;

	Horizontalseg.create(OTSU2.rows, OTSU2.cols, CV_8U);




	for (int r = 0; r < OTSU2.rows; r++)
	{
		int a = 0;
		for (int c = 0; c < OTSU2.cols; c++)
		{
			if (OTSU2.at<uchar>(r, c) == 0) Horizontalseg.at<uchar>(r, a++) = 0;


		}
	}

	////imshow("Horizontalseg", Horizontalseg);


	std::vector<int> colVerHiss(Verticalseg.cols);

	for (int i = 0; i < Verticalseg.cols; i++)
	{
		for (int j = 0; j < Verticalseg.rows; j++)
		{
			if (Verticalseg.at<uchar>(j, i) != 0)
			{
				colVerHiss[i]++;
			}
		}


	}

	vector<int> Verticalrecord;          //////////////////////////////////////////////////////////// 水平投影切割

	int Vertal = 0, Veravg = 0;

	for (size_t i = 0; i < colVerHiss.size() - 1; i++)
	{
		//cout << colVerHiss[i]<<" " ;

		Vertal = colVerHiss[i];
		Veravg += Vertal;


	}
	cout << endl;

	std::vector<int> VerHiss(Verticalseg.rows);

	VerHiss = colVerHiss;
	std::sort(VerHiss.begin(), VerHiss.end(), myobject);
	int VerHisss_ses = (VerHiss.size()*0.85);
	cout << "水平投影標準: " << VerHiss[VerHisss_ses] << endl;




	Veravg = (Veravg / colVerHiss.size());
	//cout << "Veravg: " << Veravg << endl;

	for (size_t i = 0; i < colVerHiss.size() - 1; i++)
	{



		if (colVerHiss[i] > VerHiss[VerHisss_ses])
		{

			Verticalrecord.push_back((i));


		}

	}
	cout << endl;



	vector<int> Verticalrecord2;


	for (int i = colVerHiss.size() - 1; i > 0; i--)
	{

		if (colVerHiss[i] >Veravg)
		{

			Verticalrecord2.push_back((i));


		}
	}








	vector<int>Horizontalrecord;          //////////////////////////////////////////////////////////// 垂直投影切割

	int Horizontals = 0, Horizontalavg = 0;

	std::vector<int> colHorizontalHiss(Horizontalseg.rows);

	for (int i = 0; i < Horizontalseg.rows; i++)
	{
		for (int j = 0; j < Horizontalseg.cols; j++)
		{
			if (Horizontalseg.at<uchar>(i, j) != 0)
			{
				colHorizontalHiss[i]++;
			}
		}


	}


	for (size_t i = 0; i < colHorizontalHiss.size() - 1; i++)
	{


		Horizontals = colHorizontalHiss[i];
		Horizontalavg += Horizontals;


	}
	cout << endl;

	std::vector<int> HorizontalHiss(Horizontalseg.rows);

	HorizontalHiss = colHorizontalHiss;
	std::sort(HorizontalHiss.begin(), HorizontalHiss.end(), myobject);
	int HorizontalHiss_ses = (HorizontalHiss.size()*0.8);
	cout << "垂直投影標準: " << HorizontalHiss[HorizontalHiss_ses] << endl;

	Horizontalavg = (Horizontalavg / colHorizontalHiss.size())*1.8;


	for (size_t i = 0; i < colHorizontalHiss.size() - 1; i++)
	{



		if (colHorizontalHiss[i] > HorizontalHiss[HorizontalHiss_ses])
		{

			Horizontalrecord.push_back((i));


		}

	}
	cout << endl;

	vector<int>Horizontalrecord2;


	for (int i = colHorizontalHiss.size() - 1; i > 0; i--)
	{

		if (colHorizontalHiss[i] >Horizontalavg)
		{

			Horizontalrecord2.push_back((i));


		}
	}

	if (Verticalrecord2.empty() || Verticalrecord.empty() || Horizontalrecord2.empty() || Horizontalrecord.empty())
	{
		cout << "vector_record is empty" << endl;

	}

	else
	{

		Mat Vertical_la = image(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));

		MatU  barcode = co_img(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));
		MatU barcode2 = bernsen(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));
		MatU barcode3 = all_OTSU(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));
		MatU barcode4 = all_adaptive(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));
		MatU barcode5 = Hist_bernsen(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));
		MatU barcode6 = linear(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));

		threshold(barcode, barcode, 150, 255, THRESH_BINARY | THRESH_OTSU);

		bitwise_not(barcode, barcode);

		MatU nbm, nbm2, nbm3, nbm4, nbm5, nbm6;

		Rotation_barcode(barcode, nbm);

		Rotation_barcode(barcode2, nbm2);
		Rotation_barcode(barcode3, nbm3);
		Rotation_barcode(barcode4, nbm4);
		Rotation_barcode(barcode5, nbm5);
		Rotation_barcode(barcode6, nbm6);

		read_barcode(nbm, nbm2, nbm3, nbm4, nbm5, nbm6, reseg, img_num);

		////imshow("Vertical_la", nbm3);
	}
}

///////////////////////////////////////////////////////////////////////////////////

void  Location_seg(Mat& image, int& reseg, int& img_num)
{
	cout << "使用投影定位切割" << endl;

	Mat grad_x, grad_y, xx;
	Mat abs_grad_x, abs_grad_y;

	Mat co_img, bernsen, all_OTSU, all_adaptive, Hist_bernsen, Gamma, linear;

	cvtColor(image, co_img, CV_RGB2GRAY);


	equalizeHist(co_img, Hist_bernsen);

	cv::Mat sharp_resultl, sharp_OTSU;


	sharpenImage1(co_img, sharp_resultl);


	bernsen = thresh_bernsen(co_img, 25, 40);

	adaptiveThreshold(co_img, all_adaptive, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 65, 0);

	threshold(sharp_resultl, all_OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);


	threshold(co_img, sharp_OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);

	Hist_bernsen = thresh_bernsen(Hist_bernsen, 25, 40);


	linearTrans(co_img, linear);
	linear = thresh_bernsen(linear, 25, 40);


	bitwise_not(bernsen, bernsen);
	bitwise_not(all_OTSU, all_OTSU);
	bitwise_not(all_adaptive, all_adaptive);
	bitwise_not(sharp_OTSU, sharp_OTSU);
	bitwise_not(Hist_bernsen, Hist_bernsen);
	bitwise_not(linear, linear);






	Sobel(co_img, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);  //轉成CV_8U     水平增強



	Sobel(co_img, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);// 垂直增強

	Mat Horizontal, vertical, OTSU, OTSU2, CLOSE;

	subtract(abs_grad_x, abs_grad_y, Horizontal);//X-Y
	subtract(abs_grad_y, abs_grad_x, vertical);//Y-X

	threshold(Horizontal, OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);//OTSU 二值化
	threshold(vertical, OTSU2, 150, 255, THRESH_BINARY | THRESH_OTSU);//OTSU 二值化

																	  ///////////////////////////////////////////////////////////////////////////////////////row直行///col橫列
	Mat Verticalseg;
	Verticalseg.create(OTSU.rows, OTSU.cols, CV_8U);


	for (int c = 0; c < OTSU.cols; c++)
	{
		int a = 0;
		for (int r = 0; r < OTSU.rows; r++)
		{

			if (OTSU.at<uchar>(r, c) == 0) Verticalseg.at<uchar>(a++, c) = 0;
		}
	}

	//imshow("Verticalseg", Verticalseg);

	Mat Horizontalseg;

	Horizontalseg.create(OTSU.rows, OTSU.cols, CV_8U);




	for (int r = 0; r < OTSU.rows; r++)
	{
		int a = 0;
		for (int c = 0; c < OTSU.cols; c++)
		{
			if (OTSU.at<uchar>(r, c) == 0) Horizontalseg.at<uchar>(r, a++) = 0;


		}
	}

	//imshow("Horizontalseg", Horizontalseg);


	std::vector<int> colVerHiss(Verticalseg.cols);

	for (int i = 0; i < Verticalseg.cols; i++)
	{
		for (int j = 0; j < Verticalseg.rows; j++)
		{
			if (Verticalseg.at<uchar>(j, i) != 0)
			{
				colVerHiss[i]++;
			}
		}


	}

	vector<int> Verticalrecord;          //////////////////////////////////////////////////////////// 水平投影切割

	int Vertal = 0, Veravg = 0;

	for (size_t i = 0; i < colVerHiss.size() - 1; i++)
	{
		//cout << colVerHiss[i]<<" " ;

		Vertal = colVerHiss[i];
		Veravg += Vertal;


	}
	cout << endl;

	std::vector<int> VerHiss(Verticalseg.rows);

	VerHiss = colVerHiss;
	std::sort(VerHiss.begin(), VerHiss.end(), myobject);
	int VerHisss_ses = (VerHiss.size()*0.8);
	cout << "水平投影標準: " << VerHiss[VerHisss_ses] << endl;




	Veravg = (Veravg / colVerHiss.size());
	//cout << "Veravg: " << Veravg << endl;

	for (size_t i = 0; i < colVerHiss.size() - 1; i++)
	{



		if (colVerHiss[i] > VerHiss[VerHisss_ses])
		{

			Verticalrecord.push_back((i));


		}

	}
	cout << endl;


	vector<int> Verticalrecord2;


	for (int i = colVerHiss.size() - 1; i > 0; i--)
	{

		if (colVerHiss[i] > Veravg)
		{

			Verticalrecord2.push_back((i));


		}
	}








	vector<int>Horizontalrecord;          //////////////////////////////////////////////////////////// 垂直投影切割

	int Horizontals = 0, Horizontalavg = 0;

	std::vector<int> colHorizontalHiss(Horizontalseg.rows);

	for (int i = 0; i < Horizontalseg.rows; i++)
	{
		for (int j = 0; j < Horizontalseg.cols; j++)
		{
			if (Horizontalseg.at<uchar>(i, j) != 0)
			{
				colHorizontalHiss[i]++;
			}
		}


	}


	for (size_t i = 0; i < colHorizontalHiss.size() - 1; i++)
	{


		Horizontals = colHorizontalHiss[i];
		Horizontalavg += Horizontals;


	}
	cout << endl;

	std::vector<int> HorizontalHiss(Horizontalseg.rows);

	HorizontalHiss = colHorizontalHiss;
	std::sort(HorizontalHiss.begin(), HorizontalHiss.end(), myobject);
	int HorizontalHiss_ses = (HorizontalHiss.size()*0.85);
	cout << "垂直投影標準: " << HorizontalHiss[HorizontalHiss_ses] << endl;

	Horizontalavg = (Horizontalavg / colHorizontalHiss.size())*1.8;


	for (size_t i = 0; i < colHorizontalHiss.size() - 1; i++)
	{



		if (colHorizontalHiss[i] >  HorizontalHiss[HorizontalHiss_ses])
		{

			Horizontalrecord.push_back((i));


		}

	}
	cout << endl;

	vector<int>Horizontalrecord2;


	for (int i = colHorizontalHiss.size() - 1; i > 0; i--)
	{

		if (colHorizontalHiss[i] > Horizontalavg)
		{

			Horizontalrecord2.push_back((i));


		}
	}
	if (Verticalrecord2.empty() || Verticalrecord.empty() || Horizontalrecord2.empty() || Horizontalrecord.empty())
	{
		cout << "vector_record is empty" << endl;

	}

	else
	{
		Mat Vertical_la = image(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));

		MatU  barcode = co_img(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));
		MatU barcode2 = bernsen(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));
		MatU barcode3 = all_OTSU(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));
		MatU barcode4 = all_adaptive(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));
		MatU barcode5 = Hist_bernsen(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));
		MatU barcode6 = linear(Rect(Verticalrecord[0], Horizontalrecord[0], (Verticalrecord2[0] - Verticalrecord[0]), (Horizontalrecord2[0]) - Horizontalrecord[0]));

		threshold(barcode, barcode, 150, 255, THRESH_BINARY | THRESH_OTSU);

		bitwise_not(barcode, barcode);

		read_barcode(barcode, barcode2, barcode3, barcode4, barcode5, barcode6, reseg, img_num);
		//imshow("Vertical_la", barcode3);
	}


	if (reseg == 0)
	{
		Location_seg2(image, reseg, img_num);

	}


}


void MyGammaCorrection(Mat& src, Mat& dst, float fGamma)
{
	CV_Assert(src.data);

	// accept only char type matrices  
	CV_Assert(src.depth() != sizeof(uchar));

	// build look up table  
	unsigned char lut[256];
	for (int i = 0; i < 256; i++)
	{
		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);
	}

	dst = src.clone();
	const int channels = dst.channels();
	switch (channels)
	{
	case 1:
	{

		MatIterator_<uchar> it, end;
		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)
			//*it = pow((float)(((*it))/255.0), fGamma) * 255.0;  
			*it = lut[(*it)];

		break;
	}
	case 3:
	{

		MatIterator_<Vec3b> it, end;
		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)
		{
			//(*it)[0] = pow((float)(((*it)[0])/255.0), fGamma) * 255.0;  
			//(*it)[1] = pow((float)(((*it)[1])/255.0), fGamma) * 255.0;  
			//(*it)[2] = pow((float)(((*it)[2])/255.0), fGamma) * 255.0;  
			(*it)[0] = lut[((*it)[0])];
			(*it)[1] = lut[((*it)[1])];
			(*it)[2] = lut[((*it)[2])];
		}

		break;

	}
	}
}

///////////////////////////////////////////////////////////////////////////////////
void imagecolor(Mat& src, Mat& dst)
{
	cv::Mat labelImage;
	cv::Mat stats, centroids;
	Mat cropedImage = src;

	int nLabels = cv::connectedComponentsWithStats(cropedImage, labelImage, stats, centroids, 8, CV_32S);/////八連通
	std::vector<cv::Vec3b> colors(nLabels);
	colors[0] = cv::Vec3b(0, 0, 0);

	int lab = 0;

	for (int label = 1; label < nLabels; ++label) {
		colors[label] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
		int labh = stats.at<int>(label, cv::CC_STAT_HEIGHT);
		int labw = stats.at<int>(label, cv::CC_STAT_WIDTH);

		//if ((labh / labw)>1)/////////////////條紋長寬比
		//{
		//	lab += 1;
		//}

	}

	//cv::Mat dst(cropedImage.size(), CV_8UC3);////上色
	for (int r = 0; r < dst.rows; ++r) {
		for (int c = 0; c < dst.cols; ++c) {
			int label = labelImage.at<int>(r, c);
			cv::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
			pixel = colors[label];
		}
	}

}


////////////////////////////////////////////////////////////////垂直定位
static void Barcode_Location2(Mat& src, Mat& image, int& reseg, int img_num)
{
	Mat grad_x, grad_y, xx;
	Mat abs_grad_x, abs_grad_y;

	Mat co_img, bernsen, all_OTSU, all_adaptive, Hist_bernsen, Gamma, linear;

	cvtColor(image, co_img, CV_RGB2GRAY);

	equalizeHist(co_img, Hist_bernsen);


	cv::Mat sharp_resultl, sharp_OTSU;


	sharpenImage1(co_img, sharp_resultl);


	bernsen = thresh_bernsen(co_img, 25, 40);


	adaptiveThreshold(co_img, all_adaptive, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 65, 0);

	threshold(sharp_resultl, all_OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);


	threshold(co_img, sharp_OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);

	Hist_bernsen = thresh_bernsen(Hist_bernsen, 25, 40);


	/*linearTrans(co_img, linear);
	linear = thresh_bernsen(linear, 25, 40);*/
	co_img.convertTo(linear, -1, 1.5, 30);
	linear = thresh_bernsen(linear, 25, 40);


	bitwise_not(bernsen, bernsen);
	bitwise_not(all_OTSU, all_OTSU);
	bitwise_not(all_adaptive, all_adaptive);
	bitwise_not(sharp_OTSU, sharp_OTSU);
	bitwise_not(Hist_bernsen, Hist_bernsen);
	bitwise_not(linear, linear);

	////imshow("bernsen", bernsen);
	Mat src2 = src;
	//GaussianBlur(src, src, Size(3, 3), 0, 0);
	//equalizeHist(src, src);
	//linearTrans(src, src);
	//src.convertTo(src, -1, 1.5, 30);

	Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);  //轉成CV_8U     水平增強



	Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);// 垂直增強

	Mat Horizontal, vertical, OTSU, OTSU2, CLOSE;

	subtract(abs_grad_x, abs_grad_y, Horizontal);//X-Y
	subtract(abs_grad_y, abs_grad_x, vertical);//Y-X

	threshold(Horizontal, OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);//OTSU 二值化
	threshold(vertical, OTSU2, 150, 255, THRESH_BINARY | THRESH_OTSU);//OTSU 二值化

	Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));//CLOSE
	morphologyEx(OTSU2, CLOSE, MORPH_CLOSE, kernel);
	Mat kernel2 = getStructuringElement(MORPH_RECT, Size(9, 9));
	morphologyEx(CLOSE, CLOSE, MORPH_CLOSE, kernel2);

	morphologyEx(CLOSE, CLOSE, MORPH_OPEN, kernel);//OPEN

	Mat kernel3 = getStructuringElement(MORPH_RECT, Size(8, 6));
	dilate(CLOSE, CLOSE, kernel3);

	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;

	vector<cv::Mat> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours(CLOSE, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);



	for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
	{

		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a > largest_area) {
			largest_area = a;
			largest_contour_index = i;                //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}

	}

	cv::Mat labelImage2;
	cv::Mat stats2, centroids2;

	vector< Location > vect_Location;    ////////////////////////////////////////////////////////////////////////////////    條碼定位


	int nLabels2 = cv::connectedComponentsWithStats(CLOSE, labelImage2, stats2, centroids2, 8, CV_32S);/////八連通

	std::vector<cv::Vec3b> colors2(nLabels2);
	colors2[0] = cv::Vec3b(0, 0, 0);
	//std::cout << "Number of connected components = " << nLabels2 << std::endl << std::endl;

	for (int label = 1; label < nLabels2; ++label) {
		colors2[label] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
		/*std::cout << "Component " << label << std::endl;
		std::cout << "CC_STAT_AREA   = " << stats2.at<int>(label, cv::CC_STAT_AREA) << std::endl;*/

		Location Locate(stats2.at<int>(label, cv::CC_STAT_AREA), stats2.at<int>(label, cv::CC_STAT_LEFT), stats2.at<int>(label, cv::CC_STAT_TOP), stats2.at<int>(label, cv::CC_STAT_WIDTH), stats2.at<int>(label, cv::CC_STAT_HEIGHT));
		vect_Location.push_back(Locate);


		vect_Location.push_back(Locate);


	}



	sort(vect_Location.begin(), vect_Location.end());//////////////////////////////////// 排序位置  找最大面積


	int Location_y = (vect_Location[0].fifth) / 12;

	int new_Location_h = (vect_Location[0].fifth + Location_y);


	if (vect_Location[0].third + vect_Location[0].fifth + Location_y > src.rows)
	{
		new_Location_h = src.rows - vect_Location[0].third;
	}

	Mat OTSU__Location;
	threshold(src2, OTSU__Location, 150, 255, THRESH_BINARY | THRESH_OTSU);
	OTSU__Location = OTSU__Location(Rect(vect_Location[0].second, vect_Location[0].third, vect_Location[0].fourth, new_Location_h)).clone();

	MyGammaCorrection(src2, Gamma, 1.5);
	Mat bernsen_Location = thresh_bernsen(Gamma, 25, 40);

	Mat Max_Location = src(Rect(vect_Location[0].second, vect_Location[0].third, vect_Location[0].fourth, new_Location_h)).clone();

	Mat linear_Location = linear(Rect(vect_Location[0].second, vect_Location[0].third, vect_Location[0].fourth, new_Location_h)).clone();

	bernsen_Location = bernsen_Location(Rect(vect_Location[0].second, vect_Location[0].third, vect_Location[0].fourth, new_Location_h)).clone();



	threshold(Max_Location, Max_Location, 150, 255, THRESH_BINARY | THRESH_OTSU);

	bitwise_not(OTSU__Location, OTSU__Location);

	bitwise_not(bernsen_Location, bernsen_Location);

	bitwise_not(Max_Location, Max_Location);


	/*Mat bernsen;                                                        /////////////////////////////////////////////////////////   thresh_bernsen

	bernsen = thresh_bernsen(Max_Location, 15, 15);
	bitwise_not(bernsen, Max_Location);

	//imshow("bernsen", Max_Location);*/


	Mat new_barcode;

	Rotation_barcode(Max_Location, new_barcode);
	Rotation_barcode(OTSU__Location, OTSU__Location);
	Rotation_barcode(bernsen_Location, bernsen_Location);
	Rotation_barcode(Max_Location, new_barcode);

	//imshow("new_barcode", new_barcode);


	cv::Mat cc(image.size(), CV_8UC3);////上色
	for (int r = 0; r < cc.rows; ++r) {
		for (int c = 0; c < cc.cols; ++c) {
			int label = labelImage2.at<int>(r, c);
			cv::Vec3b &pixel = cc.at<cv::Vec3b>(r, c);
			pixel = colors2[label];
		}
	}



	Scalar color(255, 255, 255);
	//drawContours(dst, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.
	rectangle(image, bounding_rect, Scalar(255, 0, 255), 2, 8, 0);

	MatU  barcode = src(bounding_rect).clone();
	MatU barcode2 = bernsen(bounding_rect).clone();
	MatU barcode3 = all_OTSU(bounding_rect).clone();
	MatU barcode4 = all_adaptive(bounding_rect).clone();
	MatU barcode5 = Hist_bernsen(bounding_rect).clone();
	MatU barcode6 = linear(bounding_rect).clone();
	/*MatU  barcode5;
	sharpenImage1(barcode, barcode5);*/


	//equalizeHist(barcode, barcode);
	//cv::addWeighted(barcode, 1.5, barcode, -0.5, 0, barcode);
	Mat copyToImg;
	barcode.copyTo(copyToImg);
	threshold(barcode, barcode, 150, 255, THRESH_BINARY | THRESH_OTSU);
	//adaptiveThreshold(barcode, barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 65, 0);
	bitwise_not(barcode, barcode);







	/*cout << "條紋:" << lab << " " << endl;
	if (lab> 29 )
	{
	cout << "通過條紋檢查" << endl;
	read_barcode(barcode);
	test += 1;
	}*/

	//read_barcode(barcode);

	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy2;
	Mat edge;
	Rect bounding_rect2;
	Canny(barcode, edge, 50, 150, 3);
	findContours(edge, contours2, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	vector<int> selected;






	////////////////////////////////////////////////////////////////////////////////////////傾斜校正

	MatU nbm, nbm2, nbm3, nbm4, nbm5, nbm6;

	Rotation_barcode(barcode, nbm);

	Rotation_barcode(barcode2, nbm2);
	Rotation_barcode(barcode3, nbm3);
	Rotation_barcode(barcode4, nbm4);
	Rotation_barcode(barcode5, nbm5);
	Rotation_barcode(barcode6, nbm6);


	//read_barcode(nbm, nbm2, nbm3, nbm4, nbm5, nbm6, reseg, img_num);
	//sec_read_barcode( nbm2, nbm3, nbm4, nbm5, nbm6, reseg, img_num);
	//third_read_barcode( nbm3, nbm4, nbm5, nbm6, reseg, img_num);
	//four_read_barcode( nbm4, nbm5, nbm6, reseg, img_num);
	//five_read_barcode( nbm5, nbm6, reseg, img_num);
	//six_read_barcode( nbm6, reseg, img_num);


	////imshow("src", src);
	////imshow("OTSU2", OTSU2);
	////imshow("image", image);
	//////imshow("abs_grad_x", abs_grad_x);
	//////imshow("img", img);
	////imshow("vertical", vertical);
	////imshow("label", cc);
	////imshow("barcode", barcode);
	////imshow("edge", edge);

	////imshow("nbm", nbm);
	////imshow("copyToImg", copyToImg);






	Mat cropedImage = barcode;

	/////////////////////////////////////////////////////////////////////////////////////////////////////






	seg_barcode(new_barcode);

	/////////////////////////////////////////////////////////////////////////////////////////數字分割


	cv::Mat dig = cv::imread("onlydigitImage3.jpg", CV_LOAD_IMAGE_GRAYSCALE);


	threshold(dig, dig, 150, 255, THRESH_BINARY | THRESH_OTSU);



	/*int rr = (dig.rows);

	IplImage* dimg;
	*dimg = IplImage(dig);*/

	//number(dig);
	//CutNum(dimg,rr);

	//svm_digitsort(dig, bernsen_Location, OTSU__Location, linear_Location, img_num);
	digitsort(dig, bernsen_Location, OTSU__Location, linear_Location, img_num);
	//xor_digitsort(dig, bernsen_Location, OTSU__Location, linear_Location, img_num);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	////imshow("digit", dig);



	if (reseg == 0)
	{
		Location_seg(image, reseg, img_num);

	}




}

////////////////////////////////////////////////////////////////////水平定位
static void Barcode_Location(Mat& src, Mat& image, int& reseg, int img_num)
{
	Mat grad_x, grad_y, xx;
	Mat abs_grad_x, abs_grad_y;

	Mat co_img, bernsen, all_OTSU, all_adaptive, Hist_bernsen, Gamma, linear;


	cvtColor(image, co_img, CV_RGB2GRAY);

	/*MyGammaCorrection(co_img, Gamma,1.5);
	Gamma = thresh_bernsen(Gamma, 25, 40);
	bitwise_not(Gamma, Gamma);
	//imshow("Gamma", Gamma);*/


	equalizeHist(co_img, Hist_bernsen);

	cv::Mat sharp_resultl, sharp_OTSU;


	sharpenImage1(co_img, sharp_resultl);


	bernsen = thresh_bernsen(co_img, 25, 40);

	adaptiveThreshold(co_img, all_adaptive, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 65, 0);

	threshold(sharp_resultl, all_OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);


	threshold(co_img, sharp_OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);

	Hist_bernsen = thresh_bernsen(Hist_bernsen, 25, 40);


	//linearTrans(co_img, linear);

	co_img.convertTo(linear, -1, 1.5, 30);
	linear = thresh_bernsen(linear, 25, 40);


	bitwise_not(bernsen, bernsen);
	bitwise_not(all_OTSU, all_OTSU);
	bitwise_not(all_adaptive, all_adaptive);
	bitwise_not(sharp_OTSU, sharp_OTSU);
	bitwise_not(Hist_bernsen, Hist_bernsen);
	bitwise_not(linear, linear);
	////imshow("bernsen", bernsen);

	//src.convertTo(src, -1, 1.5, 30);
	//linearTrans(src, src);
	////imshow("Gamma", Gamma);

	Mat src2 = src;

	GaussianBlur(src, src, Size(3, 3), 0, 0);
	//medianBlur(src, src, 3);

	//equalizeHist(src, src);


	Sobel(src, grad_x, CV_16S, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);  //轉成CV_8U     水平增強



	Sobel(src, grad_y, CV_16S, 0, 1, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(grad_y, abs_grad_y);// 垂直增強

	Mat Horizontal, vertical, OTSU, OTSU2, CLOSE;

	subtract(abs_grad_x, abs_grad_y, Horizontal);//X-Y
	subtract(abs_grad_y, abs_grad_x, vertical);//Y-X

	threshold(Horizontal, OTSU, 150, 255, THRESH_BINARY | THRESH_OTSU);//OTSU 二值化
	threshold(vertical, OTSU2, 150, 255, THRESH_BINARY | THRESH_OTSU);//OTSU 二值化







	Mat kernel = getStructuringElement(MORPH_RECT, Size(7, 7));//CLOSE
	morphologyEx(OTSU, CLOSE, MORPH_CLOSE, kernel);
	Mat kernel2 = getStructuringElement(MORPH_RECT, Size(9, 9));
	morphologyEx(CLOSE, CLOSE, MORPH_CLOSE, kernel2);

	morphologyEx(CLOSE, CLOSE, MORPH_OPEN, kernel);//OPEN

	Mat kernel3 = getStructuringElement(MORPH_RECT, Size(6, 8));
	dilate(CLOSE, CLOSE, kernel3);

	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;

	RotatedRect bounding_angle;

	vector<cv::Mat> contours; // Vector for storing contour
	vector<Vec4i> hierarchy;

	findContours(CLOSE, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);



	for (int i = 0; i < contours.size(); i++) // iterate through each contour. 
	{

		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a > largest_area) {
			largest_area = a;
			largest_contour_index = i;                //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
			bounding_angle = minAreaRect(contours[i]);
		}

	}

	float blob_angle = bounding_angle.angle;

	cout << "blob_angle_deg: " << blob_angle << endl;


	cv::Mat labelImage2;
	cv::Mat stats2, centroids2;

	vector< Location > vect_Location;    ////////////////////////////////////////////////////////////    條碼定位


	int nLabels2 = cv::connectedComponentsWithStats(CLOSE, labelImage2, stats2, centroids2, 8, CV_32S);/////八連通

	std::vector<cv::Vec3b> colors2(nLabels2);
	colors2[0] = cv::Vec3b(0, 0, 0);
	//std::cout << "Number of connected components = " << nLabels2 << std::endl << std::endl;

	for (int label = 1; label < nLabels2; ++label) {
		colors2[label] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
		//::cout << "Component " << label << std::endl;
		//std::cout << "CC_STAT_AREA   = " << stats2.at<int>(label, cv::CC_STAT_AREA) << std::endl;

		Location Locate(stats2.at<int>(label, cv::CC_STAT_AREA), stats2.at<int>(label, cv::CC_STAT_LEFT), stats2.at<int>(label, cv::CC_STAT_TOP), stats2.at<int>(label, cv::CC_STAT_WIDTH), stats2.at<int>(label, cv::CC_STAT_HEIGHT));
		vect_Location.push_back(Locate);


	}



	sort(vect_Location.begin(), vect_Location.end());//////////////////////////////////// 排序位置  找最大面積


													 /*for (int i = 0; i < vect_Location.size(); i++)
													 {
													 cout << "(" << vect_Location[i].first << "," << vect_Location[i].second << "," << vect_Location[i].third << "," << vect_Location[i].fourth << "," << vect_Location[i].fifth << ")\n";
													 }*/


	int Location_x = (vect_Location[0].fourth) / 12;

	int new_Location_w = (vect_Location[0].fourth + Location_x);
	int new_Location_x = (vect_Location[0].second - Location_x);

	cout << "new_Location_w:" << new_Location_w << endl;

	if (new_Location_x < 0)
	{
		new_Location_x = 0;
	}

	if (vect_Location[0].second + vect_Location[0].fourth + Location_x > src.cols)
	{
		new_Location_w = src.cols - vect_Location[0].second;
	}

	Mat OTSU__Location;
	threshold(src2, OTSU__Location, 150, 255, THRESH_BINARY | THRESH_OTSU);
	OTSU__Location = OTSU__Location(Rect(new_Location_x, vect_Location[0].third, new_Location_w, vect_Location[0].fifth)).clone();


	Mat Max_Location = src(Rect(new_Location_x, vect_Location[0].third, new_Location_w, vect_Location[0].fifth)).clone();

	MyGammaCorrection(co_img, Gamma, 1.5);

	Mat bernsen_Location = thresh_bernsen(Gamma, 25, 40);


	bernsen_Location = bernsen_Location(Rect(new_Location_x, vect_Location[0].third, new_Location_w, vect_Location[0].fifth)).clone();

	threshold(Max_Location, Max_Location, 150, 255, THRESH_BINARY | THRESH_OTSU);


	Mat linear_Location = linear(Rect(new_Location_x, vect_Location[0].third, new_Location_w, vect_Location[0].fifth)).clone();



	bitwise_not(bernsen_Location, bernsen_Location);

	bitwise_not(Max_Location, Max_Location);

	bitwise_not(OTSU__Location, OTSU__Location);

	////imshow("Max_Location", Max_Location);

	/*Mat bernsen;                                                        /////////////////////////////////////////////////////////   thresh_bernsen

	bernsen = thresh_bernsen(Max_Location, 25, 40);
	bitwise_not(bernsen, Max_Location);

	//imshow("bernsen", Max_Location);*/




	cv::Mat cc(image.size(), CV_8UC3);////上色
	for (int r = 0; r < cc.rows; ++r) {
		for (int c = 0; c < cc.cols; ++c) {
			int label = labelImage2.at<int>(r, c);
			cv::Vec3b &pixel = cc.at<cv::Vec3b>(r, c);
			pixel = colors2[label];
		}
	}








	Scalar color(255, 255, 255);
	//drawContours(dst, contours, largest_contour_index, color, CV_FILLED, 8, hierarchy); // Draw the largest contour using previously stored index.
	//rectangle(image, bounding_rect, Scalar(255, 0, 255), 2, 8, 0);

	//MatU barcode;
	////////////////////////              ///////////////////////////////////////   剪出條碼

	MatU barcode0 = image(bounding_rect).clone();
	MatU barcode = src(bounding_rect).clone();
	MatU barcode2 = bernsen(bounding_rect).clone();
	MatU barcode3 = all_OTSU(bounding_rect).clone();
	MatU barcode4 = all_adaptive(bounding_rect).clone();
	MatU barcode5 = Hist_bernsen(bounding_rect).clone();
	MatU barcode6 = linear(bounding_rect).clone();
	/*MatU  barcode5;
	sharpenImage1(barcode, barcode5);*/
	////imshow("barcode6", barcode6);
	/*resize(barcode0, barcode0, barcode.size());
	//imshow("barcode0", barcode0);*/

	Mat filter;
	cv::Mat Matbox = cv::Mat::zeros(barcode2.rows, barcode2.cols, CV_8UC1);
	cv::Mat getpixel = cv::Mat::zeros(barcode2.rows, barcode2.cols, CV_8UC1);

	int Mask[9][2] = {
		{ -1, -1 },{ 0, -1 },{ 1, -1 },
		{ -1, 0 },{ 0, 0 },{ 1, 0 },
		{ -1, 1 },{ 0, 1 },{ 1, 1 }
	};

	/*Sharpening(barcode2, Mask, Matbox);
	threshold(Matbox, Matbox, 150, 255, THRESH_BINARY | THRESH_OTSU);*/

	////imshow("barcode2", barcode2);
	////imshow("Matbox", Matbox);



	cv::Mat Lap = cv::Mat::zeros(barcode.rows, barcode.cols, CV_8UC1);


	//equalizeHist(barcode, barcode);
	//cv::addWeighted(barcode, 1.5, barcode, -0.5, 0, barcode);
	Mat copyToImg, copyToImg2, copyToImg3;
	barcode.copyTo(copyToImg);



	Laplacian(copyToImg, Lap, CV_16S, 3, 1, 0, BORDER_DEFAULT);
	convertScaleAbs(Lap, copyToImg3);  //轉成CV_8U
									   //threshold(copyToImg3, copyToImg3, 100, 255, THRESH_BINARY );
	threshold(copyToImg3, copyToImg3, 150, 255, THRESH_BINARY | THRESH_OTSU);

	MatU rbar = copyToImg3;

	double thres_val = threshold(barcode, barcode, 150, 255, THRESH_BINARY | THRESH_OTSU);
	//adaptiveThreshold(barcode, barcode, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 65, 0);


	bitwise_not(barcode, barcode);


	Mat barcodecoloe1(barcode.size(), CV_8UC3), barcodecoloe2(barcode2.size(), CV_8UC3), barcodecoloe3(barcode4.size(), CV_8UC3), barcodecoloe4(barcode5.size(), CV_8UC3), barcodecoloe5(barcode6.size(), CV_8UC3);


	/*imagecolor(barcode, barcodecoloe1);
	//imshow("barcodecoloe1", barcodecoloe1);

	imagecolor(barcode2, barcodecoloe2);
	//imshow("barcodecoloe2", barcodecoloe2);

	imagecolor(barcode4, barcodecoloe3);
	//imshow("barcodecoloe3", barcodecoloe3);

	imagecolor(barcode5, barcodecoloe4);
	//imshow("barcodecoloe4", barcodecoloe4);

	imagecolor(barcode6, barcodecoloe5);
	//imshow("barcodecoloe5", barcodecoloe5);*/




	cout << "thres_val: " << thres_val << endl;

	//////////////////////////////////////////////////////取條碼中間&數字切割

	seg_barcode(Max_Location);

	/////////////////////////////////////////////////////////////////////////////////////////數字分割


	cv::Mat dig = cv::imread("onlydigitImage3.jpg", CV_LOAD_IMAGE_GRAYSCALE);



	threshold(dig, dig, 150, 255, THRESH_BINARY | THRESH_OTSU);



	/*int rr = (dig.rows);

	IplImage* dimg;
	*dimg = IplImage(dig);*/

	//number(dig);
	//CutNum(dimg,rr);

	//svm_digitsort(dig, bernsen_Location, OTSU__Location, linear_Location, img_num);
	digitsort(dig, bernsen_Location, OTSU__Location, linear_Location, img_num);
	//xor_digitsort(dig, bernsen_Location, OTSU__Location, linear_Location, img_num);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	////imshow("digit", dig);

	//////////////////////////////////////////////////////////////////////////////////////////////
	cv::Mat labelImage;
	cv::Mat stats, centroids;
	Mat cropedImage = barcode;

	int nLabels = cv::connectedComponentsWithStats(cropedImage, labelImage, stats, centroids, 8, CV_32S);/////八連通
	std::vector<cv::Vec3b> colors(nLabels);
	colors[0] = cv::Vec3b(0, 0, 0);
	std::cout << "Number of connected components = " << nLabels << std::endl << std::endl;
	int lab = 0;

	for (int label = 1; label < nLabels; ++label) {
		colors[label] = cv::Vec3b((std::rand() & 255), (std::rand() & 255), (std::rand() & 255));
		int labh = stats.at<int>(label, cv::CC_STAT_HEIGHT);
		int labw = stats.at<int>(label, cv::CC_STAT_WIDTH);

		//if ((labh / labw)>1)/////////////////條紋長寬比
		//{
		//	lab += 1;
		//}

	}

	cv::Mat dst(cropedImage.size(), CV_8UC3);////上色
	for (int r = 0; r < dst.rows; ++r) {
		for (int c = 0; c < dst.cols; ++c) {
			int label = labelImage.at<int>(r, c);
			cv::Vec3b &pixel = dst.at<cv::Vec3b>(r, c);
			pixel = colors[label];
		}
	}

	MatU rb, rb2, rb3, rb4, rb5, rb6;

	Rotation_contour(barcode, rb, blob_angle);
	Rotation_contour(barcode2, rb2, blob_angle);
	Rotation_contour(barcode3, rb3, blob_angle);
	Rotation_contour(barcode4, rb4, blob_angle);
	Rotation_contour(barcode5, rb5, blob_angle);
	Rotation_contour(barcode6, rb6, blob_angle);



	//imshow("rb", rb);


	//////////////////////////////////////////////////////////////////////////////////////////////////
	//cout << "條紋:" << lab << " " << endl;
	if (nLabels > 10)
	{
		cout << "通過物件檢查" << endl;
		//read_barcode(barcode, barcode2, barcode3, barcode4, barcode5, barcode6, reseg, img_num);
		//sec_read_barcode( barcode2, barcode3, barcode4, barcode5, barcode6, reseg, img_num);
		//third_read_barcode( barcode3, barcode4, barcode5, barcode6, reseg, img_num);
		//four_read_barcode( barcode4, barcode5, barcode6, reseg, img_num);
		//five_read_barcode(barcode5, barcode6, reseg, img_num);
		//six_read_barcode( barcode6, reseg, img_num);

		test += 1;


		////imshow("src", src);
		////imshow("OTSU", OTSU);
		////imshow("image", image);
		//////imshow("abs_grad_x", abs_grad_x);
		//////imshow("img", img);
		////imshow("Horizontal", Horizontal);
		////imshow("label", cc);
		////imshow("barcode", barcode);
		////imshow("dst", dst);
		//////imshow("cropedImage", cropedImage);
		//////imshow("nbm", nbm);
		////imshow("copyToImg", copyToImg3);

		if (reseg == 0)
		{
			//read_barcode(rb, rb2, rb3, rb4, rb5, rb6, reseg, img_num);
			if (reseg == 0)
			{

				Location_seg(image, reseg, img_num);
			}
		}


	}
	else
	{
		Barcode_Location2(src, image, reseg, img_num);

	}
	//read_barcode(barcode);

	vector<vector<Point>> contours2;
	vector<Vec4i> hierarchy2;
	Mat edge;
	Rect bounding_rect2;
	Canny(barcode, edge, 50, 150, 3);
	findContours(edge, contours2, hierarchy2, CV_RETR_TREE, CV_CHAIN_APPROX_NONE);
	vector<int> selected;






	////////////////////////////////////////////////////////////////////////////////////////傾斜校正
	/*std::vector<cv::Vec4i> lines;
	cv::Size bsize = barcode.size();
	cv::HoughLinesP(barcode, lines, 1, CV_PI / 180, 100, bsize.width / 2.f, 20);


	cv::Mat disp_lines(bsize, CV_8UC1, cv::Scalar(0, 0, 0));
	double angle = 0.;
	unsigned nb_lines = lines.size();
	for (unsigned i = 0; i < nb_lines; ++i)
	{
	cv::line(disp_lines, cv::Point(lines[i][0], lines[i][1]),
	cv::Point(lines[i][2], lines[i][3]), cv::Scalar(255, 0, 0));
	angle += atan2((double)lines[i][3] - lines[i][1],
	(double)lines[i][2] - lines[i][0]);
	}
	angle /= nb_lines; // mean angle, in radians.

	float r = angle * 180 / CV_PI;
	cout << "angle:" << r << endl;

	Mat nbm;

	cv::Point2f center = Point(barcode.cols / 2.0, barcode.rows / 2.0);

	double scale = 1.0;

	Mat rot_mat = getRotationMatrix2D(center, r, scale);
	cv::Rect bbox = cv::RotatedRect(center, barcode.size(), r).boundingRect();
	rot_mat.at<double>(0, 2) += bbox.width / 2.0 - center.x;
	rot_mat.at<double>(1, 2) += bbox.height / 2.0 - center.y;
	warpAffine(barcode, nbm, rot_mat, bbox.size());*/


	//waitKey(0);


}



///////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	double START, END;//////////計算時間
	START = clock();

	vector<int> only_digits_success, number_success, bar_success, digits_success_number, all_fail, ground_truth_fail, ground_truth_fail2;

	vector<int> one_read_bar, two_read_bar, three_read_bar, four_read_bar, five_read_bar, six_reading_bar;

	int img_num;

	for (int i = 1; i <= 100; i++)
	{

		sprintf(filename, "barcode%d.jpg", i);

		Mat src;
		Mat img = imread(filename, 1);
		if (!img.data) break;

		cvtColor(img, src, CV_BGR2GRAY);

		Mat image = imread(filename, 1);
		if (!image.data) break;

		//Mat dst(src.rows, src.cols, CV_8U, Scalar::all(0));//all black
		/*Mat eh;
		equalizeHist(src, eh);
		//imshow("window1", eh);*/


		cout << filename << " " << endl;

		img_num = i;

		Barcode_Location(src, image, reseg, img_num);

		if (onltsuc1 == 0 && onltsuc2 == 1)
		{
			only_digits_success.push_back(i);

		}


		if (onltsuc1 == 1)
		{
			bar_success.push_back(i);

		}


		if (onltsuc2 == 1)
		{
			number_success.push_back(i);

		}

		if (digits_success == 1)
		{
			digits_success_number.push_back(i);
		}

		if (onltsuc1 == 0 && onltsuc2 == 0)
		{
			all_fail.push_back(i);

		}

		if (ground_truth == 1)
		{
			ground_truth_fail.push_back(i);
		}



		if (ground_truth_digits == 1)
		{
			ground_truth_fail2.push_back(i);
		}

		//////////////////////////////////////////////////////////////////////////
		if (one_read == 1)
		{
			one_read_bar.push_back(i);
		}

		if (two_read == 1)
		{
			two_read_bar.push_back(i);
		}

		if (three_read == 1)
		{
			three_read_bar.push_back(i);
		}

		if (four_read == 1)
		{
			four_read_bar.push_back(i);
		}

		if (five_read == 1)
		{
			five_read_bar.push_back(i);
		}

		if (six_reading == 1)
		{
			six_reading_bar.push_back(i);
		}

		digits_success = 0;
		onltsuc1 = 0;
		onltsuc2 = 0;
		ground_truth = 0;
		bar_or_digits = 0;
		ground_truth_digits = 0;
		reseg = 0;

		one_read = 0, two_read = 0, three_read = 0, four_read = 0, five_read = 0, six_reading = 0;
		/*cv::Mat d1 = cv::imread("p1.jpg", 1);

		deal(d1, digit_xor);
		cout << "digit_xor: " << digit_xor << endl;*/

		cout << endl;
		cout << "//////////////////////////////////////////////////////////" << endl;
		cout << endl;
		//cout << endl << "進行運算所花費的時間：" << (END - START) / CLOCKS_PER_SEC << " S" << endl;
	}

	END = clock();


	cout << "條紋解碼成功 ：";
	for (int i = 0; i < bar_success.size(); i++) {
		cout << bar_success[i] << " ";

	}
	cout << endl;
	cout << endl;

	cout << "數字解碼成功 ：";
	for (int i = 0; i < number_success.size(); i++) {
		cout << number_success[i] << " ";

	}
	cout << endl;
	cout << endl;

	cout << "數字多對 ：";
	for (int i = 0; i < only_digits_success.size(); i++) {
		cout << only_digits_success[i] << " ";

	}


	cout << endl;
	cout << endl;

	cout << "全部解碼失敗 ：";
	for (int i = 0; i < all_fail.size(); i++) {
		cout << all_fail[i] << " ";

	}


	cout << endl;
	cout << endl;


	cout << "進入13碼辨識 ：";
	for (int i = 0; i < digits_success_number.size(); i++) {
		cout << digits_success_number[i] << " ";

	}


	cout << endl;
	cout << endl;


	cout << "條紋解碼結果與答案不相符 ：";
	for (int i = 0; i < ground_truth_fail.size(); i++) {
		cout << ground_truth_fail[i] << " ";

	}


	cout << endl;
	cout << endl;

	cout << "數字辨識結果與答案不相符 ：";
	for (int i = 0; i < ground_truth_fail2.size(); i++) {
		cout << ground_truth_fail2[i] << " ";

	}


	cout << endl;
	cout << endl;
	cout << "//////////////////////////////////////////////////////////" << endl;
	cout << endl;

	cout << "第一次前處理條紋解碼成功: " << one_read_bar.size() << "\t" << "第二次前處理條紋解碼成功: " << two_read_bar.size() << "\t" << "第三次前處理條紋解碼成功: "
		<< three_read_bar.size() << "\t" << "第四次前處理條紋解碼成功: " << four_read_bar.size() << "\t" << "第五次前處理條紋解碼成功: " << five_read_bar.size() << "\t"
		<< "第六次前處理條紋解碼成功: " << six_reading_bar.size() << endl;

	cout << endl;

	cout << "第一次前處理條紋解碼成功 ：";
	for (int i = 0; i < one_read_bar.size(); i++) {
		cout << one_read_bar[i] << " ";

	}
	cout << endl;
	cout << endl;

	cout << "第二次前處理條紋解碼成功 ：";
	for (int i = 0; i < two_read_bar.size(); i++) {
		cout << two_read_bar[i] << " ";

	}
	cout << endl;
	cout << endl;

	cout << "第三次前處理條紋解碼成功 ：";
	for (int i = 0; i < three_read_bar.size(); i++) {
		cout << three_read_bar[i] << " ";

	}
	cout << endl;
	cout << endl;

	cout << "第四次前處理條紋解碼成功 ：";
	for (int i = 0; i < four_read_bar.size(); i++) {
		cout << four_read_bar[i] << " ";

	}
	cout << endl;
	cout << endl;

	cout << "第五次前處理條紋解碼成功 ：";
	for (int i = 0; i < five_read_bar.size(); i++) {
		cout << five_read_bar[i] << " ";

	}
	cout << endl;
	cout << endl;
	cout << "第六次前處理條紋解碼成功 ：";
	for (int i = 0; i < six_reading_bar.size(); i++) {
		cout << six_reading_bar[i] << " ";

	}
	cout << endl;
	cout << endl;
	cout << "//////////////////////////////////////////////////////////" << endl;


	cout << endl << "程式執行所花費：" << (double)clock() / CLOCKS_PER_SEC << " S" << endl;
	cout << "通過條紋測試數量: " << test << endl;
	cout << "第一次成功解碼數量: " << one_success << "\t" << "第二次成功解碼數量: " << two_success << "\t" << "第三次成功解碼數量: " << three_success << "\t" << "第四次成功解碼數量: " << four_success << endl;
	cout << "全部成功解碼數量: " << success << endl;
	cout << "13碼解碼數量: " << digits_success_number.size() << endl;
	cout << "數字多對數量: " << only_digits_success.size() << endl;
	cout << "數字成功解碼數量: " << digits2_success << endl;
	cout << "全部解碼失敗數量: " << all_fail.size() << endl;
	waitKey(0);
	system("pause");
	return 0;
	
}


