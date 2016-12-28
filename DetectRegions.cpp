#include "DetectRegions.h"

void DetectRegions::setFilename(string s)
{
	filename = s;
}

DetectRegions::DetectRegions()
{
	showSteps = false;
	saveRegions = false;
}

//判断矩形是否合符要求
bool DetectRegions::verifySizes (RotatedRect candidate)
{
	float error = 0.4;
	//Spain car plate size: 52*11 aspect 4,7272
	const float aspect = 4.7272;
	//Set a min and max area, All other patches are discared
	int min = 15 * aspect * 15; 
	int max = 125 *aspect * 125;
	//Get only patches that match to a respect ratio.
	float rmin = aspect - aspect * error;
	float rmax = aspect + aspect* error;

	int area =candidate.size.height * candidate.size.width;

	float r = (float) candidate.size.width / (float) candidate.size.height;
	if(r  < 1)
		r = 1/r;
	if ((area < min || area > max) || (r < rmin || r > rmax))
		return false;
	else
		return true;
}


Mat DetectRegions::histeq(Mat in)
{
	Mat out (in.size(), in.type());
	if (in.channels() == 3)
	{
		Mat hsv;
		vector < Mat > HsvSplit;
		cvtColor(in, hsv, CV_BGR2HSV);
		split(hsv,HsvSplit);
		equalizeHist(HsvSplit[2], HsvSplit[2]);
		merge(HsvSplit, hsv);
		cvtColor(hsv, out,	CV_HSV2BGR);
	}
	else if(in.channels() == 1)
		equalizeHist(in, out);
	return out;
}

vector <Plate> DetectRegions::segment(Mat input)
{
	vector <Plate> output;
	
	Mat img_gray;
	//change color mode RGB to Gray mode
	cvtColor(input, img_gray, CV_BGR2GRAY);
	//使用滤波函数blur，对输入图像进行均值滤波
	blur(img_gray, img_gray, Size(5,5));

	Mat img_sobel;
	//sobel算子进行边缘检测
	Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	if(showSteps)
		imshow("Sobel",img_sobel);

	Mat img_threshold;
	//图像二值化处理，使用threshold函数
	threshold(img_sobel, img_threshold, 0, 255,	CV_THRESH_OTSU + CV_THRESH_BINARY);
	if(showSteps)
		imshow("Thershold",img_threshold);
	//取得结构元素
	Mat element = getStructuringElement(MORPH_RECT, Size(17,3));
	//morphologyEx函数是opencv中利用图像膨胀和腐蚀的手段来执行更高级的形态学变换，例如开闭运算
	//开运算：先膨胀再腐蚀的操作，效果：消除小物体、在纤细点处分离物体、平滑较大物体的边界的同时并不明显改变其面积
	//闭运算：先膨胀后腐蚀的过程，排除小型黑洞(黑色区域)
	morphologyEx(img_threshold,img_threshold, CV_MOP_CLOSE, element);
	if(showSteps)
		imshow("Close", img_threshold);

	vector < vector < Point> > contours;
	//查找轮廓，CV_RETR_EXTERNAL查找外轮廓，CV_CHAIN_APPROX_NONE逼近函数，将所有点由链码形式翻译(转化）为点序列形式
	findContours(img_threshold, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	vector < vector < Point > > :: iterator itc = contours.begin();
	vector < RotatedRect > rects;
	//对轮廓进行遍历
	while (itc != contours.end())
	{
		//RotatedRect可旋转的矩形，opencv3寻找最小包围矩形-minAreaRect函数 
		//在轮廓得到的点中寻找一个能包括他们的矩形
		RotatedRect mr = minAreaRect(Mat (*itc));
		//初步筛出一些不合逻辑的矩形轮廓
		if( !verifySizes(mr))
			//earse函数删除指定的元素
			itc = contours.erase(itc);
		else
		{
			++itc;
			rects.push_back(mr);
		}

	}
	
	Mat result;
	input.copyTo(result);
	//画出轮廓
	drawContours(result, contours, -1, Scalar(255, 0, 0), 1);

	//For better rect cropping for each posible box
	//Make floodfill algorithm because the plate has white background
	//And then we can retrieve more clearly the contour box
	for (int i = 0; i < rects.size(); i++)
	{
		//Scalar(0,255,0)circle有三通道，通道一的值为0，通道二的值为255，等等。这里的意思是RGB值为0 255 0的值
		circle( result, rects[i].center, 3, Scalar(0, 255, 0), -1);
		float minSize = (rects[i].size.width < rects[i].size.height)? rects[i].size.width : rects[i].size.height;
		minSize = minSize - minSize * 0.5;

		//initialize rand and get 5 points around center for floodfill algorithm
		srand( time (NULL));

		//Initialize floodfill parameters and variables，RGB：0 0 0为黑色
		Mat mask;
		mask.create(input.rows + 2, input.cols + 2, CV_8UC1);
		mask = Scalar::all(0);
		int loDiff = 30;//seed与周围的像素点的负差最大值
		int upDiff = 30;//seed与周围的像素点的正差最大值
		int connectivity = 4;//连接参数
		int newMaskVal = 255;
		int NumSeeds = 10;
		Rect ccomp;
		int flags = connectivity + (newMaskVal << 8 ) + CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;
		for (int j = 0; j < NumSeeds; j++)
		{
			Point seed;
			seed.x = rects[i].center.x + rand() % (int) minSize - (minSize / 2);
			seed.y = rects[i].center.y + rand() % (int) minSize - (minSize / 2);
			//画圆函数circle(picture,center,r,Scalar(0,0,0));承载图像，圆心，半径，线的颜色RGB
			circle(result, seed, 1, Scalar(0,255,255), -1);
			//floodFill函数水漫算法，填充出中重点处理的部分
			int area = floodFill(input, mask, seed, Scalar(255, 0, 0), &ccomp,Scalar(loDiff,loDiff,loDiff), Scalar(upDiff,upDiff,upDiff), flags );

		}
		if(showSteps)
			imshow("MASK", mask);

		vector < Point> pointsInterest;
		Mat_<uchar>::iterator itMask = mask.begin<uchar>();
		Mat_<uchar>::iterator end = mask.end<uchar>();

		for (; itMask != end; ++itMask)
		{
			if(*itMask == 255)
				pointsInterest.push_back(itMask.pos());
		}
		RotatedRect minRect = minAreaRect(pointsInterest);

		if (verifySizes(minRect))
		{
			// rotated rectangle drawing
			// 画出初步识别的矩形 
			Point2f rect_points[4];
			minRect.points( rect_points);
			for( int j = 0; j < 4; j++)
				line( result, rect_points[j], rect_points[(j+1) % 4], Scalar(0, 0, 255), 1, 8);

			//Get rotation matrix
			float r = (float) minRect.size.width / (float) minRect.size.height;
			float angle = minRect.angle;
			if(r < 1)
				angle = angle + 90;
			//已知旋转中心坐标（坐标原点为图像左上端点）、旋转角度（单位为度°，顺时针为负，逆时针为正）、放缩比例，返回旋转/放缩矩阵
			Mat rotmat = getRotationMatrix2D(minRect.center, angle, 1);


			//Create and rotate image
			Mat img_rotated;
			//执行图像旋转，CV_INTER_CUBIC插值函数，三次插值
			warpAffine(input, img_rotated, rotmat, input.size(), CV_INTER_CUBIC);


			//Crop image
			Size rect_size = minRect.size;
			if(r < 1 )
				swap(rect_size.width, rect_size.height);
			Mat img_crop;
			getRectSubPix(img_rotated, rect_size, minRect.center, img_crop);

			Mat resultResized;
			resultResized.create(33, 144, CV_8UC3);
			resize(img_crop, resultResized, resultResized.size(), 0, 0, INTER_CUBIC);

			//Equalize croped image
			Mat grayResult;
			cvtColor(resultResized, grayResult, CV_BGR2GRAY);
			blur(grayResult, grayResult, Size(3,3));
			grayResult = histeq(grayResult);
			if (saveRegions)
			{
				stringstream ss (stringstream::in | stringstream ::out);
				ss << filename << "temp"<< "_" << i << ".jpg";
				imwrite(ss.str(), result);
			}
			output.push_back(Plate (grayResult, minRect.boundingRect()));
		}
	}
	if (showSteps)
	{
		imshow("Contours",result);
	}

	cvWaitKey(0);
	return output;

}

vector <Plate> DetectRegions :: run (Mat input)
{
	//Segement image by white
	vector<Plate> tmp = segment(input);

	return tmp;
}
