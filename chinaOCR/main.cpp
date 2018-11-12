#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<fstream>
#include<ml.h>
#include "feature.h"
using namespace cv;
using namespace std;
using namespace ml;
bool verifySize(RotatedRect mr)
{
	float error = 0.47;//
	float aspect = 3.75;//3.1429
	float rmin = aspect*(1 - error);
	float rmax = aspect*(1 + error);
	float min = 15 * aspect * 15;
	float max = 125 * aspect * 125;
	float area = mr.size.height*mr.size.width;
	float r = mr.size.width / mr.size.height;
	float flag = r;
	if (r < 1)
		r = mr.size.height / mr.size.width;
	if ((area<min || area>max) || (r<rmin || r>rmax)|| (flag>1&&(mr.angle<-10&&mr.angle>-90))||(flag<1&&(mr.angle>-80&&mr.angle<-0))||(flag<1&&mr.angle==0) )
		return false;
	else
		return true;

	/*if ((area>min &&area<max) && (r>rmin && r<rmax)  )
		return true;
	else
		return false;*/
}

void main()
{
	bool show =false;
	string imgname;
	string savename;
	string file_name;
	ifstream fin("C:/Users/root/Desktop/train IMG/test.txt");
	Mat src;//= imread("C:/Users/root/Desktop/train IMG/浙GZJ021.jpg");
	int sum = 136;
	while (getline(fin, imgname))
	{
		sum--;
		cout << "剩余图片个数：" << sum << endl;
		file_name = imgname;
		file_name.erase(file_name.end()-4,file_name.end());
		imgname = "C:/Users/root/Desktop/train IMG/" + imgname;
		src = imread(imgname);
		Mat gray;
		GaussianBlur(src, gray, Size(3, 3), 0, 0);//9时，会使字符E虑的不成形。
		cvtColor(gray, gray, CV_BGR2GRAY);
		Mat img_sobel, img_abs;
		Sobel(gray, img_sobel, CV_8U, 1, 0, 3);//-1
		convertScaleAbs(img_sobel, img_abs);
		Mat grad;
		addWeighted(img_abs, 1, 0, 0, 0, grad);
		//imshow("sobel", grad);/////
		Mat img_threshold;
		threshold(grad, img_threshold, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);
		/*imshow("threshold", img_threshold);*/
		Mat element = getStructuringElement(MORPH_RECT, Size(17, 5));
		morphologyEx(img_threshold, img_threshold, CV_MOP_CLOSE, element);
		if(show==true)
		imshow("闭运算", img_threshold);

		Mat img_erode, img_dilate;
		Mat erode_element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));//5
		erode(img_threshold, img_erode, erode_element, Point(-1, -1), 3);

		Mat dilate_element = getStructuringElement(MORPH_ELLIPSE, Size(7, 3));//8,6,//开始为4,(6,3),(8,4),(7,3),(8,8)
		dilate(img_erode, img_dilate, dilate_element, Point(-1, -1), 4);//
		if (show == true)
		imshow("多次腐蚀膨胀", img_dilate);


		vector<vector<Point>> con;
		findContours(img_dilate, con, RETR_EXTERNAL, CHAIN_APPROX_NONE);//img_threshold

		vector<vector<Point>>::iterator itc = con.begin();
		vector<RotatedRect> rects;

		while (itc != con.end())
		{
			RotatedRect mr = minAreaRect(Mat(*itc));
			if (!verifySize(mr))
				itc = con.erase(itc);
			else
			{
				rects.push_back(mr);
				++itc;
			}
		}
		Point2f rect_points[4];
		Mat code;
		cout << rects.size() << endl;
		vector<Mat> img_posible;//测试中
		for (int i = 0; i < rects.size(); i++)
		{
			Mat temp = src.clone();
			
			/*cout << "width--" << i << "-:" << rects[i].size.width << endl;
			cout << "height--" << i << "-:" << rects[i].size.height << endl;
			cout << "angle--" << i << "-:" << rects[i].angle << endl;*/
			rects[i].points(rect_points);
			if (show == true)
			{
				for (int j = 0; j < 4; j++)
				{
					line(src, rect_points[j], rect_points[(j + 1) % 4], Scalar(0, 0, 255));
				}
			}
			stringstream ss;
			ss << i;
			float r = rects[i].size.width / rects[i].size.height;
			float angle = rects[i].angle;
			if (r < 1)
			{
				angle = 90 + angle;
				swap(rects[i].size.width, rects[i].size.height);
			}
			Mat rotmat = getRotationMatrix2D(rects[i].center, angle, 1);
			Mat img_rotated;
			warpAffine(temp, img_rotated, rotmat, temp.size(), INTER_CUBIC);
			//imshow("旋转后" + ss.str(), img_rotated);
			cout << "width:" << rects[i].size.width << "--height:" << rects[i].size.height << "---:" << rects[i].size << endl;
			getRectSubPix(img_rotated, rects[i].size, rects[i].center, code);
			if (show == true)
			imshow("截取的区域" + ss.str(), code);

			Mat img_resize;
			img_resize.create(36, 136, CV_8UC3);
			resize(code, img_resize, img_resize.size(), 0, 0, INTER_CUBIC);

			Mat grayresult;
			cvtColor(img_resize, grayresult, CV_BGR2GRAY);
			GaussianBlur(grayresult, grayresult, Size(3, 3), 0, 0);
			grayresult = histeq(grayresult);
			//imshow("调整大小" + ss.str(), grayresult);
			/*savename = "C:/Users/root/Desktop/SVM/" + file_name + "_" + to_string(i) + ".jpg";
			imwrite(savename,grayresult);*/			
			img_posible.push_back(grayresult);
		}
		if (show == true)
		imshow("可能为车牌的区域", src);

		
		Ptr<SVM> svm_pre = SVM::load("C:/Users/root/Desktop/chinaOCR/trainSVM/SVM_feature_2.xml");		
		vector<Mat> plate;
		for (int i=0;i<img_posible.size();i++)
		{			
			Mat img = img_posible[i];
			Mat feature;
			getHistogramFeatures(img, feature);
			Mat p = feature.reshape(1, 1);
			p.convertTo(p,CV_32FC1);
			cout<<"svm predict---:"<<svm_pre->predict(p)<<endl;
			int response = (int)svm_pre->predict(p);
			if (response == 1)
				plate.push_back(img_posible[i]);
		}
		/*for (int i = 0; i < plate.size(); i++)
		{			
			savename = "C:/Users/root/Desktop/SVM/test_2/" + file_name + "_" + to_string(i) + ".jpg";
			imwrite(savename, plate[i]);
			cout << "显示分类结果：" << endl;
		}*/
	}
	waitKey(0);
}