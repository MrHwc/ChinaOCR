#ifndef _FEATURE_H_
#define _FEATURE_H_
#include "opencv2/opencv.hpp"

using namespace cv;


Mat ProjectedHistogram(Mat img, int t, int threshold = 20);
Mat getHistogram(cv::Mat in);
void getHistogramFeatures(const cv::Mat& image, cv::Mat& features);
Mat histeq(Mat in);
Mat charFeatures(Mat in, int sizeData);
#endif
