#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<fstream>
#include "feature.h"
using namespace cv;
using namespace std;

float countOfBigValue(Mat &mat, int iValue) 
{
	float iCount = 0.0;
	if (mat.rows > 1) {
		for (int i = 0; i < mat.rows; ++i) {
			if (mat.data[i * mat.step[0]] > iValue) {
				iCount += 1.0;
			}
		}
		return iCount;

	}
	else {
		for (int i = 0; i < mat.cols; ++i) {
			if (mat.data[i] > iValue) {
				iCount += 1.0;
			}
		}

		return iCount;
	}
}

Mat ProjectedHistogram(Mat img, int t, int threshold)// threshold
{
	int sz = (t) ? img.rows : img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j < sz; j++) {
		Mat data = (t) ? img.row(j) : img.col(j);

		mhist.at<float>(j) = countOfBigValue(data, threshold);//直方图，统计>threshold的个数
	}

	// Normalize histogram
	double min, max;
	minMaxLoc(mhist, &min, &max);

	if (max > 0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);

	return mhist;
}

Mat getHistogram(Mat in) 
{
	const int VERTICAL = 0;
	const int HORIZONTAL = 1;

	// Histogram features
	Mat vhist = ProjectedHistogram(in, VERTICAL);
	Mat hhist = ProjectedHistogram(in, HORIZONTAL);

	// Last 10 is the number of moments components
	int numCols = vhist.cols + hhist.cols;

	Mat out = Mat::zeros(1, numCols, CV_32F);

	int j = 0;
	for (int i = 0; i < vhist.cols; i++) {
		out.at<float>(j) = vhist.at<float>(i);
		j++;
	}
	for (int i = 0; i < hhist.cols; i++) {
		out.at<float>(j) = hhist.at<float>(i);
		j++;
	}

	return out;
}


void getHistogramFeatures(const Mat& image, Mat& features) {
	/*Mat grayImage;
	cvtColor(image, grayImage, CV_RGB2GRAY);*/

	//grayImage = histeq(grayImage);

	Mat img_threshold;
	threshold(image, img_threshold, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	//Mat img_threshold = grayImage.clone();
	//spatial_ostu(img_threshold, 8, 2, getPlateType(image, false));

	features = getHistogram(img_threshold);
}

//直方图均衡化
Mat histeq(Mat in)
{
	Mat out(in.size(), in.type());
	if (in.channels() == 3)
	{
		Mat hsv;
		vector<Mat> hsvSplit;
		cvtColor(in, hsv, CV_BGR2GRAY);
		split(hsv, hsvSplit);
		equalizeHist(hsvSplit[2], hsvSplit[2]);
		merge(hsvSplit, hsv);
		cvtColor(hsv, out, CV_HSV2BGR);
	}
	else if (in.channels() == 1)
	{
		equalizeHist(in, out);
	}
	return out;
}