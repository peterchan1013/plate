#ifndef OCR_h
#define OCR_h

#include <string.h>
#include <vector>

#include <cv.h>
#include <highgui.h>
#include <cvaux.h>
#include <ml.h>
#include "Plate.h"


using namespace std;
using namespace cv;

#define HORIZONTAL 1
#define VERTICAL 0


class CharSegment
{
public:
	CharSegment();
	CharSegment(Mat i, Rect p);
	Mat img;
	Rect pos;
	
};


class OCR
{
public:
	bool DEBUG;
	bool saveSegments;
	string filename;
	static const int numCharacters;
	static const char strCharacters[];

	OCR(string trainFile);
	OCR();
	string run(Plate *input);
	int charSize;
	Mat preprocessChar (Mat in);
	int classify(Mat f);
	void train(Mat TrainData, Mat trianClasses, int nlayers);
	int classifyKnn(Mat f);
	void trainKnn(Mat TrainSamples, Mat trianClasses, int k);
	Mat features(Mat input, int size);

private:
	bool trained;
	vector <CharSegment> segement (Plate input);
	Mat Preprocess(Mat in, int newSize);
	Mat getVisualHistogram(Mat *hist, int type);
	void drawVisualFeatures(Mat character, Mat hhist, Mat vhist, Mat lowData);
	Mat ProjectedHistogram(Mat img, int i);
	bool verifySizes(Mat r);
	CvANN_MLP ann;
	CvKNearest knnClassifier;
	int K;

};


#endif