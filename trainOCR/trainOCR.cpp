#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<fstream>
#include<ml.h>
#include<string>
#include "feature.h"
using namespace cv;
using namespace std;
using namespace ml;
const char *strCharacters[] = { "0","1","2","3","4","5","6","7","8","9",
								"A","B","C","D","E","F","G","H","J","K","L","M",
								"N","P","Q","R","S","T","U","V","W","X","Y","Z",
								 "沪", "苏", "浙", "京"};//"hu", "su", "zhe", "jing"
//vector<Mat> horizontalProjectionMat(Mat srcImg)//水平投影
//{
//	Mat binImg;
//	blur(srcImg, binImg, Size(3, 3));
//	threshold(binImg, binImg, 0, 255, CV_THRESH_OTSU);
//	int perPixelValue = 0;//每个像素的值
//	int width = srcImg.cols;
//	int height = srcImg.rows;
//	int* projectValArry = new int[height];//创建一个储存每行白色像素个数的数组
//	memset(projectValArry, 0, height * 4);//初始化数组
//	for (int col = 0; col < height; col++)//遍历每个像素点
//	{
//		for (int row = 0; row < width; row++)
//		{
//			perPixelValue = binImg.at<uchar>(col, row);
//			if (perPixelValue == 0)//如果是白底黑字
//			{
//				projectValArry[col]++;
//			}
//		}
//	}
//	Mat horizontalProjectionMat(height, width, CV_8UC1);//创建画布
//	for (int i = 0; i < height; i++)
//	{
//		for (int j = 0; j < width; j++)
//		{
//			perPixelValue = 255;
//			horizontalProjectionMat.at<uchar>(i, j) = perPixelValue;//设置背景为白色
//		}
//	}
//	for (int i = 0; i < height; i++)//水平直方图
//	{
//		for (int j = 0; j < projectValArry[i]; j++)
//		{
//			perPixelValue = 0;
//			horizontalProjectionMat.at<uchar>(i, width - 1 - j) = perPixelValue;//设置直方图为黑色
//		}
//	}
//	vector<Mat> roiList;//用于储存分割出来的每个字符
//	int startIndex = 0;//记录进入字符区的索引
//	int endIndex = 0;//记录进入空白区域的索引
//	bool inBlock = false;//是否遍历到了字符区内
//	for (int i = 0; i <srcImg.rows; i++)
//	{
//		if (!inBlock && projectValArry[i] != 0)//进入字符区
//		{
//			inBlock = true;
//			startIndex = i;
//		}
//		else if (inBlock && projectValArry[i] == 0)//进入空白区
//		{
//			endIndex = i;
//			inBlock = false;
//			Mat roiImg = srcImg(Range(startIndex, endIndex + 1), Range(0, srcImg.cols));//从原图中截取有图像的区域
//			roiList.push_back(roiImg);
//		}
//	}
//	delete[] projectValArry;
//	return roiList;
//}


vector<Mat> verticalProjectionMat(Mat srcImg,Mat binImg)//垂直投影
{
		
	int perPixelValue;//每个像素的值
	int width = srcImg.cols;
	int height = srcImg.rows;
	int* projectValArry = new int[width];//创建用于储存每列白色像素个数的数组
	memset(projectValArry, 0, width * 4);//初始化数组
	for (int col = 0; col < width; col++)
	{
		for (int row = 0; row < height; row++)
		{
			perPixelValue = binImg.at<uchar>(row, col);
			if (perPixelValue == 255)//如果是白底黑字//0
			{
				projectValArry[col]++;
			}
		}
	}

	Mat verticalProjectionMat(height, width, CV_8UC1);//垂直投影的画布
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			perPixelValue = 0;  //背景设置为白色//255
			verticalProjectionMat.at<uchar>(i, j) = perPixelValue;
		}
	}
	for (int i = 0; i < width; i++)//垂直投影直方图
	{
		for (int j = 0; j < projectValArry[i]; j++)
		{
			perPixelValue = 255;  //直方图设置为黑色  //0
			verticalProjectionMat.at<uchar>(height - 1 - j, i) = perPixelValue;
		}
	}
	imshow("垂直投影", verticalProjectionMat);
	cvWaitKey(0);

	vector<Mat> roiList;//用于储存分割出来的每个字符
	int startIndex = 0;//记录进入字符区的索引
	int endIndex = 0;//记录进入空白区域的索引
	bool inBlock = false;//是否遍历到了字符区内
	for (int i = 0; i < srcImg.cols; i++)//cols=width
	{
		if (!inBlock && projectValArry[i] != 0)//进入字符区
		{
			inBlock = true;
			startIndex = i;
		}
		else if (projectValArry[i] == 0 && inBlock)//进入空白区
		{
			endIndex = i;
			inBlock = false;
			Mat roiImg = srcImg(Range(0, srcImg.rows), Range(startIndex, endIndex + 1));
			roiList.push_back(roiImg);
		}
	}
	delete[] projectValArry;
	return roiList;
}

bool verifyCharSizes(Mat r) 
{
	// Char sizes 45x90
	float aspect = 45.0f / 90.0f;
	float charAspect = (float)r.cols / (float)r.rows;
	float error = 0.65f;//0.7//0.47//0.49//0.65
	float minHeight = 10.f;
	float maxHeight = 35.f;
	// We have a different aspect ratio for number 1, and it can be ~0.2
	float minAspect = 0.05f;
	float maxAspect = aspect + aspect * error;
	// area of pixels
	int area = cv::countNonZero(r);
	// bb area
	int bbArea = r.cols * r.rows;
	//% of pixel in area
	int percPixels = area / bbArea;

	/*if (percPixels <= 1 && charAspect > minAspect && charAspect < maxAspect &&
		r.rows >= minHeight && r.rows < maxHeight)
		return true;*/
	if (charAspect > minAspect && charAspect < maxAspect&&bbArea>40&& area>35&& r.rows >=16)//40,35,16//
		return true;
	else
		return false;
}

Mat preprocessChar(Mat in) 
{
	// Remap image
	int h = in.rows;
	int w = in.cols;

	int charSize = 20;

	Mat transformMat = Mat::eye(2, 3, CV_32F);
	int m = max(w, h);
	transformMat.at<float>(0, 2) = float(m / 2 - w / 2);
	transformMat.at<float>(1, 2) = float(m / 2 - h / 2);

	Mat warpImage(m, m, in.type());
	warpAffine(in, warpImage, transformMat, warpImage.size(), INTER_LINEAR,
		BORDER_CONSTANT, Scalar(0));

	Mat out;
	resize(warpImage, out, Size(charSize, charSize));

	return out;
}
int char_classify(Mat f, Ptr<cv::ml::ANN_MLP> ann)
{	
	Mat output(1, 34, CV_32FC1);//numCharacters
	ann->predict(f, output);
	Point maxLoc;
	double maxVal;
	minMaxLoc(output, 0, &maxVal, 0, &maxLoc);
	//We need know where in output is the max val, the x (cols) is the class.
	return maxLoc.x;
}
int Chinese_classify(Mat f, Ptr<cv::ml::ANN_MLP> ann)
{	
	Mat output(1, 4, CV_32FC1);//numCharacters
	ann->predict(f, output);
	Point maxLoc;
	double maxVal;
	minMaxLoc(output, 0, &maxVal, 0, &maxLoc);
	//We need know where in output is the max val, the x (cols) is the class.
	return maxLoc.x;
}

Mat convertTo3Channels(const Mat& binImg)
{
	Mat three_channel = Mat::zeros(binImg.rows, binImg.cols, CV_8UC3);
	vector<Mat> channels;
	for (int i = 0; i<3; i++)
	{
		channels.push_back(binImg);
	}
	merge(channels, three_channel);
	return three_channel;
}
void main()//分割字符
{
	string imgname;
	string savename;
	string file_name;
	ifstream fin("C:/Users/root/Desktop/new/Plates/plates.txt");	
	Mat src;	
	/*while (getline(fin, imgname))
		{		
		file_name = imgname;
		file_name.erase(file_name.end() - 4, file_name.end());
		imgname = "C:/Users/root/Desktop/new/Plates/" + imgname;*/
		
			src = imread("C:/Users/root/Desktop/new/Plates/A03_A722S6_1.jpg");//	imgname
			//Mat img_src = src(Rect_<double>(src.size().width*0.028, src.size().height*0.067, src.size().width*0.95, src.size().height*0.86));
			Mat gray;		
			cvtColor(src, gray, CV_BGR2GRAY);
			Mat img_threshold;				
			threshold(gray, img_threshold, 0, 255, CV_THRESH_BINARY + CV_THRESH_OTSU);// //187						
			//imshow("threshold", img_threshold);
			
			vector<vector<Point>> con;
			findContours(img_threshold,con, RETR_EXTERNAL, CHAIN_APPROX_NONE);
			
			vector<vector<Point>>::iterator itc = con.begin();
			vector<Rect> rects;
			while (itc != con.end())
			{
				Rect mr = boundingRect(Mat(*itc));
				Mat auxROI(img_threshold,mr);			
				if (verifyCharSizes(auxROI))
					rects.push_back(mr);							
				++itc;
			}	
		
			if (rects.size() == 7)//得到7个字符，调整汉字
			{
				for (int i = 0; i < 6; i++)
				{
					for (int j = 0; j < 6 - i; j++)
					{
						if (rects[j].x > rects[j + 1].x)
							swap(rects[j], rects[j+1]);
					}
				}
				rects[0].y = rects[1].y;
				rects[0].x = rects[0].x*0.68;
				rects[0].height = rects[1].height;
				rects[0].width = rects[0].width*1.5;
			}
			if (rects.size() == 8)//得到8个字符，调整汉字
			{
				for (int i = 0; i < 7; i++)
				{
					for (int j = 0; j < 7 - i; j++)
					{
						if (rects[j].x > rects[j + 1].x)
							swap(rects[j], rects[j + 1]);
					}
				}
				auto it = rects.begin();
				if (rects[0].width > rects[7].width&&(abs(rects[7].y- rects[6].y)>3|| abs(rects[7].height - rects[6].height)>3))
				{
					it = rects.end()-1;
					rects.erase(it);
				}
				if (rects[0].width + 5 < rects[1].width)
				{
					it = rects.begin();
					rects.erase(it);
				}
				if (rects[0].width > 20)
				{
					it = rects.begin();
					rects.erase(it);
				}
				rects[0].y = rects[1].y;
				rects[0].x = rects[0].x*0.68;
				rects[0].height = rects[1].height;
				rects[0].width = rects[0].width*1.5;
			}
			if (rects.size() == 9)//得到9个字符，调整汉字
			{
				for (int i = 0; i < 8; i++)
				{
					for (int j = 0; j < 8 - i; j++)
					{
						if (rects[j].x > rects[j + 1].x)
							swap(rects[j], rects[j + 1]);
					}
				}
				auto it = rects.begin();
				rects.erase(it);
				it = rects.end()-1;
				rects.erase(it);
				rects[0].y = rects[1].y;
				rects[0].x = rects[0].x*0.68;
				rects[0].height = rects[1].height;
				rects[0].width = rects[0].width*1.5;
			}			
			/*for (int i = 0; i < rects.size(); i++)
			{
				rectangle(src, rects[i], Scalar(0, 0, 255));
			}
			imshow("dst", src);
			waitKey(0);*/
			/*if (rects.size() != 7)
			{
				cout<< file_name << endl;
			}*/
			//for (int i = 0; i < 7; i++)//调整字符大小，保存图片
			//{
			//	Mat ch(img_threshold, rects[i]);
			//	ch=preprocessChar(ch);
			//	savename="C:/Users/root/Desktop/new/char/"+ file_name + "_" + to_string(i) + ".jpg";
			//	imwrite(savename,ch);
			//}
			vector<Mat> vec_plates;
			vector<String> plate_number;
			for (int i = 0; i < 7; i++)
			{
				Mat ch(img_threshold, rects[i]);
				ch = preprocessChar(ch);
				vec_plates.push_back(ch);
				imshow(to_string(i), ch);
				/*savename = "C:/Users/root/Desktop/" + file_name + "_" + to_string(i) + ".jpg";
				imwrite(savename, ch);*/
			}
			
			
			for (int i = 0; i < 7; i++)
			{
				if (i == 0)
				{	
					
					Mat f1 = charFeatures(vec_plates[i], 10);
					Ptr<ANN_MLP> ann_1 = ANN_MLP::load("C:/Users/root/Desktop/data/ann_chniese_test.xml");
					int flag = char_classify(f1, ann_1);
					cout << flag << endl;
					plate_number.push_back(strCharacters[34+ flag]);// 
				}
				else
				{
					Mat temp = convertTo3Channels(vec_plates[i]);
					Mat f1 = charFeatures(temp, 10);//			vec_plates[i]
					Ptr<ANN_MLP> ann_2 = ANN_MLP::load("C:/Users/root/Desktop/data/ann_char_c3_48.xml");
					int flag = char_classify(f1, ann_2);
					cout << flag << endl;
					cout << strCharacters[flag] << endl;
					plate_number.push_back(strCharacters[flag]);		
				}
			}
			for (int i = 0; i < 7; i++)
				cout << plate_number[i];
			cout << endl;
		/*}*/
		waitKey(0);	
}


const int numCharacters = 38;
//							  0	 1   2 3  4  5  6 7  8  9  A  B  C  D  E  F  G  H   J  K L M  N  P Q R S T U V W X Y Z 沪 苏 浙 京
const int numFilesChars[] = { 47,42,69,60,8,53,59,52,74,71,56,19,28,14,18,19,14,11,10,15,8,16,11,6,7,6,5,6,6,4,7,5,7,7,50,42,40,40 };

//void main()//训练ANN
//{	
//	Mat classes;
//	Mat trainingData;
//	Mat src;
//	vector<int> trainingLabels;
//	string imgname;
//	string savename;
//	string file_name;
//	string file_path;
//	for (int i = 0; i < numCharacters; i++)
//	{
//		if (i == 0)
//		{
//		file_path="C:/Users/root/Desktop/new/char/0/";
//		file_name = "C:/Users/root/Desktop/new/char/0/0.txt";
//		}
//		if (i == 1)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/1/";
//		file_name = "C:/Users/root/Desktop/new/char/1/1.txt";
//		}
//		if (i == 2)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/2/";
//		file_name = "C:/Users/root/Desktop/new/char/2/2.txt";
//		}
//		if (i == 3)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/3/";
//		file_name = "C:/Users/root/Desktop/new/char/3/3.txt";
//		}
//		if (i == 4)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/4/";
//		file_name = "C:/Users/root/Desktop/new/char/4/4.txt";
//		}
//		if (i == 5)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/5/";
//		file_name = "C:/Users/root/Desktop/new/char/5/5.txt";
//		}
//		if (i == 6)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/6/";
//		file_name = "C:/Users/root/Desktop/new/char/6/6.txt";
//		}
//		if (i == 7)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/7/";
//		file_name = "C:/Users/root/Desktop/new/char/7/7.txt";
//		}
//		if (i == 8)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/8/";
//		file_name = "C:/Users/root/Desktop/new/char/8/8.txt";
//		}
//		if (i == 9)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/9/";
//		file_name = "C:/Users/root/Desktop/new/char/9/9.txt";
//		}
//		if (i == 10)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/A/";
//		file_name = "C:/Users/root/Desktop/new/char/A/A.txt";
//		}
//		if (i == 11)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/B/";
//		file_name = "C:/Users/root/Desktop/new/char/B/B.txt";
//		cout << "11" << endl;
//		}
//		if (i == 12)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/C/";
//		file_name = "C:/Users/root/Desktop/new/char/C/C.txt";
//		cout << "12" << endl;
//		}
//		if (i == 13)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/D/";
//		file_name = "C:/Users/root/Desktop/new/char/D/D.txt";
//		}
//		if (i == 14)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/E/";
//		file_name = "C:/Users/root/Desktop/new/char/E/E.txt";
//		}
//		if (i == 15)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/F/";
//		file_name = "C:/Users/root/Desktop/new/char/F/F.txt";
//		}
//		if (i == 16)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/G/";
//		file_name = "C:/Users/root/Desktop/new/char/G/G.txt";
//		}
//		if (i == 17)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/H/";
//		file_name = "C:/Users/root/Desktop/new/char/H/H.txt";
//		}
//		if (i == 18)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/J/";
//		file_name = "C:/Users/root/Desktop/new/char/J/J.txt";
//		}
//		if (i == 19)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/K/";
//		file_name = "C:/Users/root/Desktop/new/char/K/K.txt";
//		}
//		if (i == 20)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/L/";
//		file_name = "C:/Users/root/Desktop/new/char/L/L.txt";
//		}
//		if (i == 21)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/M/";
//		file_name = "C:/Users/root/Desktop/new/char/M/M.txt";
//		}
//		if (i == 22)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/N/";
//		file_name = "C:/Users/root/Desktop/new/char/N/N.txt";
//		}
//		if (i == 23)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/P/";
//		file_name = "C:/Users/root/Desktop/new/char/P/P.txt";
//		}
//		if (i == 24)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/Q/";
//		file_name = "C:/Users/root/Desktop/new/char/Q/Q.txt";
//		}
//		if (i == 25)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/R/";
//		file_name = "C:/Users/root/Desktop/new/char/R/R.txt";
//		}
//		if (i == 26)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/S/";
//		file_name = "C:/Users/root/Desktop/new/char/S/S.txt";
//		}
//		if (i == 27)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/T/";
//		file_name = "C:/Users/root/Desktop/new/char/T/T.txt";
//		}
//		if (i == 28)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/U/";
//		file_name = "C:/Users/root/Desktop/new/char/U/U.txt";
//		}
//		if (i == 29)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/V/";
//		file_name = "C:/Users/root/Desktop/new/char/V/V.txt";
//		}
//		if (i == 30)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/W/";
//		file_name = "C:/Users/root/Desktop/new/char/W/W.txt";
//		}
//		if (i == 31)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/X/";
//		file_name = "C:/Users/root/Desktop/new/char/X/X.txt";
//		}
//		if (i == 32)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/Y/";
//		file_name = "C:/Users/root/Desktop/new/char/Y/Y.txt";
//		}
//		if (i == 33)
//		{
//		file_path = "C:/Users/root/Desktop/new/char/Z/";
//		file_name = "C:/Users/root/Desktop/new/char/Z/Z.txt";
//		}
//		if (i == 34)
//		{
//			file_path = "C:/Users/root/Desktop/new/char/沪/";
//			file_name = "C:/Users/root/Desktop/new/char/沪/hu.txt";
//		}
//		if (i == 35)
//		{
//			file_path = "C:/Users/root/Desktop/new/char/苏/";
//			file_name = "C:/Users/root/Desktop/new/char/苏/su.txt";
//		}
//		if (i == 36)
//		{
//			file_path = "C:/Users/root/Desktop/new/char/浙/";
//			file_name = "C:/Users/root/Desktop/new/char/浙/zhe.txt";
//		}
//		if (i == 37)
//		{
//			file_path = "C:/Users/root/Desktop/new/char/京/";
//			file_name = "C:/Users/root/Desktop/new/char/京/jing.txt";
//		}
//		ifstream fin(file_name);
//		while (getline(fin,imgname))
//		{
//			imgname = file_path + imgname;
//			src = imread(imgname);
//			Mat f10 = charFeatures(src, 10);
//			trainingData.push_back(f10);
//			trainingLabels.push_back(i);
//			cout << "loading--" << endl;
//		}
//		file_name = "";
//		cout << "continue-" << endl;
//	}
//	cout << "完成标签" << endl;
//	trainingData.convertTo(trainingData,CV_32FC1);
//	cv::Mat train_classes =cv::Mat::zeros((int)trainingLabels.size(), 34, CV_32F);//
//
//	for (int i = 0; i < train_classes.rows; ++i) {
//		train_classes.at<float>(i, trainingLabels[i]) = 1.f;
//	}
//	Mat layers;
//	int input_number = 120;
//	int hidden_number = 40;
//	int output_number = 34;//
//	
//	layers.create(1, 3, CV_32SC1);
//	layers.at<int>(0) = input_number;
//	layers.at<int>(1) = hidden_number;
//	layers.at<int>(2) = output_number;
//
//	Ptr<cv::ml::ANN_MLP> ann_ = ANN_MLP::create();
//	ann_->setLayerSizes(layers);
//	ann_->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM, 1, 1);
//	ann_->setTrainMethod(cv::ml::ANN_MLP::TrainingMethods::BACKPROP);
//	ann_->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 8000, 0.0001));//
//	ann_->setBackpropWeightScale(0.1);
//	ann_->setBackpropMomentumScale(0.1);
//	Ptr<cv::ml::TrainData >traindata =TrainData::create(trainingData, cv::ml::SampleTypes::ROW_SAMPLE,train_classes);
//	ann_->train(traindata);
//	ann_->save("C:/Users/root/Desktop/chinaOCR/trainOCR/ann.xml");
//
//}

//void main()
//{
//	string imgname;
//	string savename;
//	string file_name;
//	string file_path;
//	Mat src=imread("C:/Users/root/Desktop/_0.jpg");//C:/Users/root/Desktop/ann/result1/京/_0.jpg
//	cout << src.channels() << endl;
//	cout << src.depth() << endl;
//	cout << src.type() << endl;
//	/*ifstream fin("C:/Users/root/Desktop/ann/test3/ch.txt");
//	cout << "test" << endl;
//	while (getline(fin,imgname))
//	{		
//		file_name = imgname;
//		file_name.erase(file_name.end() - 4, file_name.end());
//		imgname = "C:/Users/root/Desktop/ann/test3/" + imgname;*/
//	/*	src = imread(imgname);*/
//		Mat f1 = charFeatures(src, 10);
//		Ptr<ANN_MLP> ann = ANN_MLP::load("C:/Users/root/Desktop/data/ann_chniese_1.xml");//ann_3
//		int flag = char_classify(f1, ann);
//		cout << flag << endl;
//		cout << strCharacters[flag+34] << endl;
//		//if (flag==0)//strChniese[flag]=="hu"
//		//{
//		//	cout << "complete---" << endl;
//		//	savename = "C:/Users/root/Desktop/ann/result1/沪/" + file_name + ".jpg";
//		//	imwrite(savename, src);
//		//	cout << "complete---" << endl;
//		//}
//		//if (flag == 1)//strChniese[flag]=="su"
//		//{
//		//	savename = "C:/Users/root/Desktop/ann/result1/苏/" + file_name + ".jpg";
//		//	imwrite(savename, src);
//		//}
//		//if (flag == 2)//strChniese[flag]=="zhe"
//		//{
//		//	savename = "C:/Users/root/Desktop/ann/result1/浙/" + file_name + ".jpg";
//		//	imwrite(savename, src);
//		//}
//		//if (flag == 3)//strChniese[flag]=="jing"
//		//{
//		//	savename = "C:/Users/root/Desktop/ann/result1/京/" + file_name + ".jpg";
//		//	imwrite(savename, src);
//		//}
//		/*if (strCharacters[flag] == "0")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/0/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "1")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/1/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "2")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/2/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "3")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/3/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "4")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/4/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "5")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/5/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "6")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/6/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "7")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/7/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "8")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/8/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "9")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/9/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "A")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/A/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "B")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/B/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "C")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/C/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "D")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/D/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "E")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/E/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "F")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/F/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "G")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/G/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "H")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/H/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "J")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/J/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "K")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/K/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "L")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/L/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "M")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/M/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "N")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/N/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "P")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/P/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "Q")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/Q/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "R")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/R/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "S")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/S/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "T")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/T/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "U")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/U/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "V")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/V/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "W")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/W/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "X")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/X/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "Y")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/Y/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}
//		if (strCharacters[flag] == "Z")
//		{
//			savename = "C:/Users/root/Desktop/ann/result1/Z/" + file_name + ".jpg";
//			imwrite(savename, src);
//		}*/
//	/*}*/
//	
//}