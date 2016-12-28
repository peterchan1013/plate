#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <ml.h>

#include <iostream>
#include <vector>

#include "DetectRegions.h"
#include "OCR.h"

using namespace std;
using namespace cv;

string getFilename(string s)
{
	char sep = '/';
	char sepExt = '.';

#ifdef _WIN32
	sep = '\\';
#endif

	size_t i = s.rfind(sep, s.length());
	if (i != string :: npos)
	{
		string fn = (s.substr(i + 1, s.length() - i));
		size_t j = fn.rfind(sepExt, fn.length());
		if(i != string::npos)
			return fn.substr(0,j);
		else
		    return fn;
	}
	else
		return "";

}

int main(int argc, char **argv)
{
	cout << "Opencv Automatic Number Plate Recognition\n";

	Mat input_image;
	input_image = imread("/home/hxy/hxy/final_project/plate/car.jpg");

	//string filename_whitoutExt = getFilename(filename);
	string filename_whitoutExt = "/home/hxy/hxy/final_project/plate/";
	cout << "working with file: " << filename_whitoutExt <<"\n";

	//车牌检测
	DetectRegions detectRegions;
	detectRegions.setFilename(filename_whitoutExt);
	detectRegions.saveRegions = true;
	detectRegions.showSteps = true;
	vector < Plate > posible_regions = detectRegions.run(input_image);

	/*分类车牌号与非车牌号*/
	FileStorage fs;
	fs.open("/home/hxy/hxy/final_project/plate/SVM.xml",FileStorage::READ);
    //
	Mat SVM_TrainningData;
	Mat SVM_Classes;

	fs ["TrainingData"] >> SVM_TrainningData;
	fs ["classes"] >> SVM_Classes;

	//设置SVM参数
	CvSVMParams SVM_params;
	SVM_params.kernel_type = CvSVM ::LINEAR;
	SVM_params.degree = 0;
	SVM_params.gamma = 1;
	SVM_params.coef0 = 0;
	SVM_params.C = 1;
	SVM_params.nu = 0;
	SVM_params.p = 0;
	SVM_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);

	//创建并训练分类器
	CvSVM svmClassifier (SVM_TrainningData, SVM_Classes, Mat(), Mat(), SVM_params);

	//分类车牌
	vector < Plate > plates;
	for (int i = 0; i < posible_regions.size(); i++)
	{
		Mat img = posible_regions[i].plateImg;
		Mat p = img.reshape(1, 1); // convert img to 1 row m features
		p.convertTo(p, CV_32FC1);
		int response = (int)svmClassifier.predict( p );


		if(response == 1)
			plates.push_back(posible_regions[i]);
	}

	cout << "Num plates detected: " << plates.size() << "\n";


	OCR ocr("OCR.xml");
	ocr.saveSegments = true;
	ocr.DEBUG = false;
	ocr.filename = filename_whitoutExt;
	for (int i = 0; i < plates.size(); i++)
	{
		Plate plate = plates[i];

		string plateNumber = ocr.run(&plate);
		string licensePlate = plate.str();
		cout << "==============================================\n";
		cout << " License plate number: " << licensePlate << "\n";
		cout << "==============================================\n";
		rectangle(input_image, plate.position, Scalar(0, 0, 200));
		putText (input_image, licensePlate, Point(plate.position.x, plate.position.y), CV_FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 200), 2);

		if (false)
		{
			imshow("Plate detected seg", plate.plateImg);
			cvWaitKey(0);
		}
	}


	imshow("Plate detected", input_image);
	for (;;)
	{
		int c;
		c = cvWaitKey(10);
		if((char) c == 27)
			break;
	}
	return 0;
}
