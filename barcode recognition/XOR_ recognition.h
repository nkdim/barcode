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

using namespace std;
using namespace cv;




void deal(Mat &src, int& digit_xor);
double compare(Mat &src, Mat &sample, int digit_xor);
void Threshold(Mat &src, Mat &sample, int m, int digit_xor);

