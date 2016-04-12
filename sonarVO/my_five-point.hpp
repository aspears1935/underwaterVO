#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

Mat findEssentialMatNew( InputArray _points1, InputArray _points2, double focal_x, double focal_y, Point2d pp, int method, double prob, double threshold, OutputArray _mask);

int recoverPoseNew( InputArray E, InputArray _points1, InputArray _points2, OutputArray _R, OutputArray _t, double focal_x, double focal_y, Point2d pp, InputOutputArray _mask);
