//#include "_modelest.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

//HAD TO REMOVE THIS FOR VERSION 3.0.0
//int estimateTranslation(InputArray _from, OutputArray _out, OutputArray _inliers, double param1, double param2);

void rigidTransform2D(const int N);
