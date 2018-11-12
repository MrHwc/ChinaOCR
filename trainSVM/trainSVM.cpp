#include<cv.h>
#include<highgui.h>
#include<cvaux.h>
#include<iostream>
#include<vector>
#include<fstream>
#include<ml.h>
#include "feature.h"
using namespace cv;
using namespace std;
using namespace ml;
void main()
{	
	
	Mat classes;// (435, 1, CV_32FC1);
	Mat trainingData;// (435, 4896, CV_32FC1);

	Mat trainingImages;
	vector<int> trainingLabels;

	string imgname;
	ifstream fin("C:/Users/root/Desktop/SVM/Plates/plates.txt");//C:/Users/root/Desktop/new/Plates/plates.txt
	while (getline(fin, imgname))
	{
		imgname ="C:/Users/root/Desktop/SVM/Plates/" +imgname;
		Mat img_1 = imread(imgname,0);//"C:/Users/root/Desktop/SVM/Plates/´¨A762ZS_0.jpg"
		Mat feature_1;
		getHistogramFeatures(img_1,feature_1);
		feature_1 = feature_1.reshape(1,1);
		trainingImages.push_back(feature_1);
		trainingLabels.push_back(1);
	}
	string imgname_2;
	ifstream fin_2("C:/Users/root/Desktop/SVM/NoPlates/Noplates.txt");//C:/Users/root/Desktop/new/Noplates/Noplates.txt
	while (getline(fin_2,imgname_2))
	{
		imgname_2= "C:/Users/root/Desktop/SVM/NoPlates/"+imgname_2;
		Mat img_2 = imread(imgname_2, 0);//"C:/Users/root/Desktop/SVM/NoPlates/´¨A88888_0.jpg"
		Mat feature_2;
		getHistogramFeatures(img_2, feature_2);
		feature_2 = feature_2.reshape(1, 1);
		trainingImages.push_back(feature_2);				
		trainingLabels.push_back(0);
	}

	Mat(trainingImages).copyTo(trainingData);	
	trainingData.convertTo(trainingData,CV_32FC1);
	Mat(trainingLabels).copyTo(classes);
	/*classes.convertTo(classes,CV_32SC1);*/
	/*FileStorage fs("SVM_3.xml",FileStorage::WRITE);
	fs << "TrainingData" << trainingData;
	fs << "classes" << classes;
	fs.release();*/
		
	Ptr<SVM> svm_pre = SVM::create();
	svm_pre->setType(SVM::C_SVC);
	svm_pre->setKernel(SVM::RBF);//
	svm_pre->setDegree(0.1);//
	svm_pre->setGamma(0.1);//
	svm_pre->setCoef0(0.1);//
	svm_pre->setC(1);
	svm_pre->setNu(0.1);//
	svm_pre->setP(0.1);//
	svm_pre->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER,4000,0.0001));
	//SVM_TrainingData.convertTo(SVM_TrainingData,CV_32FC1);
	//SVM_Classes.convertTo(SVM_Classes, CV_32SC1);
	Ptr<TrainData> tdata = TrainData::create(trainingData,SampleTypes::ROW_SAMPLE, classes);
	//svm_pre->train(tdata);	
	svm_pre->trainAuto(tdata, 10, SVM::getDefaultGrid(SVM::C),
		SVM::getDefaultGrid(SVM::GAMMA), SVM::getDefaultGrid(SVM::P),
		SVM::getDefaultGrid(SVM::NU), SVM::getDefaultGrid(SVM::COEF),
		SVM::getDefaultGrid(SVM::DEGREE), true);
	svm_pre->save("C:/Users/root/Desktop/chinaOCR/trainSVM/SVM_feature_2.xml");//
	waitKey(0);
}