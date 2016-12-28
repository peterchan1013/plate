#include "OCR.h"

const char OCR::strCharacters[] = {'0', '1', '2', '3','4','5','6','7','8','9','B','C','D','F','G','H','J','K','L','M','N','P','R','S','T','V','W','X','Y','Z'};
const int OCR::numCharacters = 30;

CharSegment::CharSegment()
{

}

CharSegment::CharSegment(Mat i, Rect p)
{
	img = i;
	pos = p;
}

OCR::OCR()
{
	DEBUG = false;
	trained = false;
	saveSegments = false;
	charSize = 20;
}

OCR::OCR(string trainFile)
{
	DEBUG = false;
	trained = false;
	saveSegments = false;
	charSize = 20;

	//��OCR.xml�ж�ȡѵ������
	Mat classes;
	Mat trainingData;


	FileStorage fs;
	fs.open("/home/hxy/hxy/final_project/plate/OCR.xml",FileStorage::READ);
	fs [ "TrainingDataF10"] >> trainingData;
	fs ["classes"]>> classes;

	train(trainingData, classes, 12); //ѵ��������,ʹ��10�����ز���Ԫ


}

Mat OCR::preprocessChar(Mat in)  //�����ַ�
{
	int h = in.rows;
	int w = in.cols;
	Mat trainsfromMat = Mat ::eye(2,3,CV_32F);
	int m = max(w,h);
	trainsfromMat.at<float>(0,2) = m/2 - w/2;
	trainsfromMat.at<float>(1,2) = m/2 - h/2;

	Mat warpImage(m, m, in.type());
	warpAffine(in, warpImage, trainsfromMat, warpImage.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));


	Mat out;
	resize(warpImage, out, Size(charSize, charSize));

	return out;
}

bool OCR::verifySizes(Mat r)  //�ж��ǲ����ַ�����
{
	float aspect = 45.0f / 77.0f; //�ַ��Ŀ�߱�Ϊ 45/77
	float charAspect = (float) r.cols / (float) r.rows;
	float error = 0.35;
	float minHeight = 15;
	float maxHeight = 28;

	float minAspect = 0.2;
	float maxAspect = aspect + aspect * error;
	float area = countNonZero(r); //ͳ����������
	float bbArea = r.cols * r.rows; //�������
	float percPixels = area / bbArea; //���ر�ֵ

	if(DEBUG)
		cout<< "Aspect: " << aspect << " ["<< minAspect <<"," << maxAspect <<"] "<< percPixels << " Char aspect" << charAspect << " height char"<< r.rows <<"\n";

	if(percPixels < 0.8 && charAspect > minAspect && charAspect < maxAspect && r.rows >= minHeight && r.rows < maxHeight)
		return true;
	else
		return false;
}


vector <CharSegment> OCR ::segement(Plate plate)
{
	Mat input = plate.plateImg;
	vector < CharSegment > output;
	Mat img_threshold ;
	threshold( input, img_threshold, 60, 255, CV_THRESH_BINARY_INV);
	if(DEBUG)
		imshow(" Thershold plate", img_threshold);
	Mat img_contours;
	img_threshold.copyTo(img_contours);

	vector < vector <Point> > contours;
	findContours( img_contours, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	Mat result;
	img_contours.copyTo(result);
	cvtColor(result, result, CV_GRAY2RGB);
	drawContours( result, contours, -1, Scalar(0,255,0),1);

	vector < vector <Point> > ::iterator itc = contours.begin();

	//Remove patch that are no inside limits of aspect ratio and area.
	while (itc!=contours.end())
	{
		//Create bounding rect of object
		Rect mr= boundingRect(Mat(*itc));
		rectangle(result, mr, Scalar(0,255,0));
		//Crop image
		Mat auxRoi(img_threshold, mr);
		if(verifySizes(auxRoi)){
			auxRoi=preprocessChar(auxRoi);
			output.push_back(CharSegment(auxRoi, mr));
			rectangle(result, mr, Scalar(0,125,255));
		}
		++itc;
	}

	if(DEBUG)
		cout<< "Num Chars" << output.size() << "\n";

	if(DEBUG);
		imshow("Segment chars", result);

	return output;
}


Mat OCR::ProjectedHistogram(Mat img, int t)
{
	int sz = (t) ? img.rows : img.cols;
	Mat mhist = Mat::zeros(1, sz, CV_32F);

	for (int j = 0; j < sz; j++) //���л����м������ص�
	{
		Mat data = (t)? img.row(j) : img.col(j);
		mhist.at < float > (j) = countNonZero(data);
	}

	double min, max;
	minMaxLoc(mhist, &min, &max);  //�ҳ���С�����ֵ


	if(max > 0)
		mhist.convertTo(mhist, -1, 1.0f / max, 0);   //��һ��ֱ��ͼ

	return mhist;
}

Mat OCR :: getVisualHistogram(Mat *hist, int type)
{
	int size = 100;
	Mat imHist;

	if(type == HORIZONTAL)
		imHist.create(Size(size, hist->cols), CV_8UC3);
	else
		imHist.create(Size(hist->cols, size), CV_8UC3);

	imHist = Scalar(55, 55, 55);

	for ( int i = 0; i < hist->cols; i++)
	{
		float value = hist->at <float> (i);
		int maxval = (int) (value  * size);
		Point pt1,pt2, pt3, pt4;

		if (type == HORIZONTAL)
		{
			pt1.x = pt3. x = 0;
			pt2.x = pt4.x = maxval;
			pt1.y = pt2.y = i;
			pt3.y = pt4.y = i+1;

			line(imHist, pt1, pt2, CV_RGB(220, 220, 220),1 , 8, 0);
			line(imHist, pt3, pt4, CV_RGB(34, 34, 34),1 , 8, 0);
		}
		else
		{
			pt1.x = pt2.x = i;
			pt3.x = pt4.x = i+1;
			pt1.y = pt3.y = 100;
			pt2.y = pt4.y = 100 - maxval;

			line(imHist,pt1, pt2, CV_RGB(220,220,220),1,8,0);
			line(imHist, pt3, pt4, CV_RGB(34, 34, 34),1 , 8, 0);

			pt3.x = pt4.x = i+2;
			line(imHist, pt3, pt4, CV_RGB(44, 44,44),1,8,0);

			pt3.x = pt4.x = i+3;
			line(imHist, pt3, pt4, CV_RGB(50,50,50),1,8,0);
		}
	}

	return imHist;
}


void OCR::drawVisualFeatures(Mat character, Mat hhist, Mat vhist, Mat lowData)
{
	Mat img(121, 121, CV_8UC3, Scalar(0, 0, 0));
	Mat ch;
	Mat ld;

	cvtColor(character, ch, CV_GRAY2RGB);

	resize(lowData, ld, Size(100,100), 0, 0, INTER_LINEAR); //תΪ�ͷֱ���ͼ��
	cvtColor(ld, ld, CV_GRAY2RGB);

	Mat hh = getVisualHistogram( &hhist, HORIZONTAL); //ˮƽֱ��ͼ
	Mat hv = getVisualHistogram(&hhist, VERTICAL);   //��ֱֱ��ͼ

	Mat subImg = img(Rect (0, 101, 20, 20));
	ch.copyTo(subImg);  //�ַ�ͼ��

	subImg = img(Rect(21, 101, 100, 20));
	hh.copyTo(subImg); //ˮƽֱ��ͼ

	subImg = img(Rect(0, 0, 20, 100));
	hv.copyTo(subImg); //��ֱֱ��ͼ

	subImg = img(Rect ( 21, 0, 100, 100));
	ld.copyTo(subImg);

	line(img, Point (0, 100), Point(121, 100), Scalar(0, 0, 255));
	line(img, Point(20, 0), Point(20, 121), Scalar(0,0,255));

	imshow("Visual features",img);

	cvWaitKey(0);

}

Mat OCR::features(Mat in, int sizeData) //��ȡ��������
{
	//ֱ��ͼ����
	Mat vhist = ProjectedHistogram(in, VERTICAL);
	Mat hhist = ProjectedHistogram(in, HORIZONTAL);

	//�ͷֱ���ͼ������
	Mat lowData;
	resize(in, lowData, Size(sizeData, sizeData));

	if(DEBUG)
		drawVisualFeatures(in, hhist, vhist, lowData);

	int numCols = vhist.cols + hhist.cols + lowData.cols * lowData.cols;

	Mat out = Mat :: zeros(1, numCols, CV_32F); //���������������
	int j = 0;
	for (int i =0; i < vhist.cols; i++)
	{
		out.at < float > (j) = vhist.at < float > (i);
		j++;
	}
	for (int i = 0; i < hhist.cols; i++)
	{
		out.at < float > (j) = hhist.at < float >(i);
		j++;
	}
	for (int i = 0; i < lowData.cols; i++)
	{
		for (int k = 0; k < lowData.rows; k++)
		{
			out.at < float > (j) = (float) lowData.at<unsigned char >(i,k);
			j++;
		}
	}

	if(DEBUG)
		cout<< out << "\n ======================================\n";
	return out;
}

void OCR :: train(Mat TrainData, Mat classes, int nlayers)
{
	Mat layers(1, 3, CV_32SC1);
	layers.at<int>(0) = TrainData.cols;//�����
	layers.at<int>(1) = nlayers;  //���ز�
	layers.at<int>(2) = numCharacters;  //�����

	ann.create(layers, CvANN_MLP::SIGMOID_SYM, 1, 1);

	Mat trainClasses;
	trainClasses.create(TrainData.rows, numCharacters, CV_32FC1);
	for (int i = 0; i < trainClasses.rows; i++)
	{
		for (int k = 0; k < trainClasses.cols; k++)
		{
			if(k == classes.at<int>(i))
				trainClasses.at<float>(i,k) =1;
			else
				trainClasses.at<float>(i,k) = 0;

		}
	}

	Mat weights (1,	TrainData.rows, CV_32FC1, Scalar::all(1)); //����Ȩ��

	ann.train(TrainData, trainClasses, weights); //ѵ��������


}

int OCR::classify(Mat f)
{
	int result	 = -1;
	Mat output(1, numCharacters, CV_32FC1);
	ann.predict(f, output);
	Point maxLoc;
	double maxVal;
	minMaxLoc( output, 0, &maxVal, 0, &maxLoc);

	return maxLoc.x;
}

int OCR::classifyKnn(Mat f)
{
	int response = (int) knnClassifier.find_nearest(f, K);
	return response;
}


void OCR::trainKnn(Mat TrainSamples, Mat trianClasses, int k)
{
	K = k;
	knnClassifier.train(TrainSamples, trianClasses, Mat(), false, K);

}

string OCR:: run(Plate *input)
{
	vector < CharSegment> segments = segement(*input);

	for ( int i = 0; i < segments.size(); i++)
	{
		Mat ch = preprocessChar(segments[i].img);
		if(saveSegments)
		{
			stringstream ss (stringstream :: in | stringstream ::out);
			ss  << filename <<"tempChars" << "_" << i << ".jpg";
			imwrite(ss.str(), ch);
		}
		Mat f = features(ch, 10);
		int character = classify(f);
		input->chars.push_back(strCharacters[character]);
		input->charsPos.push_back(segments[i].pos);
	}

	return "_"; //input->str();
}

