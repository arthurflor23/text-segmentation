#include "Binarization.hpp"

#define uget(x,y)at<unsigned char>(y,x)
#define uset(x,y,v)at<unsigned char>(y,x)=v;
#define fget(x,y)at<float>(y,x)
#define fset(x,y,v)at<float>(y,x)=v;

Binarization::Binarization() {};

void Binarization::binarize(Mat image, Mat &output, int option){
	cvtColor(image, this->grayscale, COLOR_BGR2GRAY);
    lightDistribution();

    int winy = (int) (2.0 * this->grayscale.rows-1)/3;
    int winx = (int) this->grayscale.cols-1 < winy ? this->grayscale.cols-1 : winy;
    if (winx > 127) winx = winy = 127;

    thresholdImg(this->grayscale, output, option, winx, winy, 0.1, 128);
}

void Binarization::thresholdImg(Mat im, Mat &output, int option, int winx, int winy, double k, double dR){

	if (option >= 3){
		Mat smoothedImg;
		blur(im, smoothedImg, Size(3,3), Point(-1,-1));
		threshold(smoothedImg, output, 0.0, 255, THRESH_BINARY | THRESH_OTSU);
		return;
	}

	double m, s, maxS;
	double th = 0;
	double minI, maxI;
	int wxh	= winx/2;
	int wyh	= winy/2;
	int xFirstth= wxh;
	int xLastth = im.cols-wxh-1;
	int yLastth = im.rows-wyh-1;
	int yFirstth = wyh;

	output = im.clone();
	Mat mapM = Mat::zeros(im.rows, im.cols, CV_32F);
	Mat mapS = Mat::zeros(im.rows, im.cols, CV_32F);
	maxS = calcLocalStats(im, mapM, mapS, winx, winy);

	minMaxLoc(im, &minI, &maxI);
	Mat thsurf(im.rows, im.cols, CV_32F);

	for	(int j=yFirstth; j<=yLastth; j++){
		float *thSurfData = thsurf.ptr<float>(j) + wxh;
		float *mapMData = mapM.ptr<float>(j) + wxh;
		float *mapSData = mapS.ptr<float>(j) + wxh;

		for	(int i=0; i<=im.cols-winx; i++) {
			m = *mapMData++;
			s = *mapSData++;

    		switch (option) {
    			case 0: // NIBLACK
    				th = m + k*s;
    				break;

    			case 1: // SAUVOLA
	    			th = m * (1 + k*(s/dR-1));
	    			break;

    			case 2: // WOLF
    				th = m + k * (s/maxS-1) * (m-minI);
    				break;
    		}
			*thSurfData++ = th;

    		if (i==0){
				float *thSurfPtr = thsurf.ptr<float>(j);
        		for (int i=0; i<=xFirstth; ++i)
					*thSurfPtr++ = th;

        		if (j==yFirstth){
        			for (int u=0; u<yFirstth; ++u){
						float *thSurfPtr = thsurf.ptr<float>(u);
						for (int i=0; i<=xFirstth; ++i)
        					*thSurfPtr++ = th;
					}
				}

        		if (j == yLastth){
        			for (int u=yLastth+1; u<im.rows; ++u){
						float *thSurfPtr = thsurf.ptr<float>(u);

        				for (int i=0; i<=xFirstth; ++i)
        					*thSurfPtr++ = th;
					}
				}
    		}

			if (j==yFirstth)
				for (int u=0; u<yFirstth; ++u)
					thsurf.fset(i+wxh,u,th);

			if (j==yLastth)
				for (int u=yLastth+1; u<im.rows; ++u)
					thsurf.fset(i+wxh,u,th);
		}
		float *thSurfPtr = thsurf.ptr<float>(j) + xLastth;

		for (int i=xLastth; i<im.cols; ++i)
			*thSurfPtr++ = th;

		if (j==yFirstth){
			for (int u=0; u<yFirstth; ++u){
				float *thSurfPtr = thsurf.ptr<float>(u) + xLastth;

				for (int i=xLastth; i<im.cols; ++i)
					*thSurfPtr++ = th;
			}
		}

		if (j==yLastth){
			for (int u=yLastth+1; u<im.rows; ++u){
				float *thSurfPtr = thsurf.ptr<float>(u) + xLastth;

				for (int i=xLastth; i<im.cols; ++i)
					*thSurfPtr++ = th;
			}
		}
	}

	for	(int y=0; y<im.rows; ++y){
		unsigned char *imData = im.ptr<unsigned char>(y);
		float *thSurfData = thsurf.ptr<float>(y);
		unsigned char *outputData = output.ptr<unsigned char>(y);

		for	(int x=0; x<im.cols; ++x){
			*outputData = *imData >= *thSurfData ? 255 : 0;
			imData++;
			thSurfData++;
			outputData++;
		}
	}
}

double Binarization::calcLocalStats(Mat &im, Mat &mapM, Mat &mapS, int winx, int winy){
    Mat imSum, imSumSq;
    integral(im, imSum, imSumSq, CV_64F);

	double m,s,maxS,sum,sumSq;
	int wxh	= winx/2;
	int wyh	= winy/2;
	int xFirstth= wxh;
    int yFirstth= wyh;
	int yLastth = im.rows-wyh-1;
	double winarea = winx*winy;

	maxS = 0;
	for	(int j = yFirstth ; j<=yLastth; j++){
		sum = sumSq = 0;

		double *sumTopLeft = imSum.ptr<double>(j - wyh);
		double *sumTopRight = sumTopLeft + winx;
		double *sumBottomLeft = imSum.ptr<double>(j - wyh + winy);
		double *sumBottomRight = sumBottomLeft + winx;

		double *sumEqTopLeft = imSumSq.ptr<double>(j - wyh);
		double *sumEqTopRight = sumEqTopLeft + winx;
		double *sumEqBottomLeft = imSumSq.ptr<double>(j - wyh + winy);
		double *sumEqBottomRight = sumEqBottomLeft + winx;

		sum = (*sumBottomRight + *sumTopLeft) - (*sumTopRight + *sumBottomLeft);
		sumSq = (*sumEqBottomRight + *sumEqTopLeft) - (*sumEqTopRight + *sumEqBottomLeft);

		m  = sum / winarea;
		s  = sqrt ((sumSq - m*sum)/winarea);
		if (s > maxS) maxS = s;

		float *mapMData = mapM.ptr<float>(j) + xFirstth;
		float *mapSData = mapS.ptr<float>(j) + xFirstth;
		*mapMData++ = m;
		*mapSData++ = s;

		for	(int i=1 ; i<=im.cols-winx; i++) {
			sumTopLeft++, sumTopRight++, sumBottomLeft++, sumBottomRight++;

			sumEqTopLeft++, sumEqTopRight++, sumEqBottomLeft++, sumEqBottomRight++;

			sum = (*sumBottomRight + *sumTopLeft) - (*sumTopRight + *sumBottomLeft);
			sumSq = (*sumEqBottomRight + *sumEqTopLeft) - (*sumEqTopRight + *sumEqBottomLeft);

			m  = sum / winarea;
			s  = sqrt ((sumSq - m*sum)/winarea);
			if (s > maxS) maxS = s;

			*mapMData++ = m;
			*mapSData++ = s;
		}
	}
	return maxS;
}

void Binarization::lightDistribution(){
	getHistogram(this->grayscale);
	getCEI();
	getEdge();
	getTLI();

    Mat intImg = this->cei.clone();

    for (int y=0; y<intImg.cols; y++){
        for (int x=0; x<intImg.rows; x++){

            if (this->tliErosion.at<float>(x,y) == 0){
                int head = x, end = x, n;

                while (end < this->tliErosion.rows && this->tliErosion.at<float>(end,y) == 0){
                    end++;
                }
                end--;
                n = end - head + 1;

                if (n <= 30){
                    vector<float> mpvH, mpvE;
                    double minH, maxH, minE, maxE;

                    for (int k=0; k<5; k++){
                        if ((head - k) >= 0)
                            mpvH.push_back(this->cei.at<float>(head-k,y));
                        if ((end + k) < this->cei.rows)
                            mpvE.push_back(this->cei.at<float>(end+k,y));
                    }

                    minMaxLoc(mpvH, &minH, &maxH);
                    minMaxLoc(mpvE, &minE, &maxE);

                    for (int m=0; m<n; m++)
                        intImg.at<float>(head+m,y) = maxH + (m+1) * ((maxE-maxH) / n);
                }
            }
        }
    }

    Mat kernel = Mat::ones(Size(11, 11), CV_32F) * 1/121;
    filter2D(scale(intImg), this->ldi, CV_32F, kernel);

    this->grayscale = (this->cei/this->ldi) * 260;

    for (int y=0; y<this->tliErosion.rows; y++){
        for (int x=0; x<this->tliErosion.cols; x++){
            if (this->tliErosion.at<float>(y,x) != 0)
                this->grayscale.at<float>(y,x) *= 1.5;
        }
    }

    GaussianBlur(this->grayscale, this->grayscale, Size(3,3), 2);
    this->grayscale.convertTo(this->grayscale, CV_8U);
}

void Binarization::getHistogram(Mat image){
    vector<Mat> bgrPlanes;
    split(image, bgrPlanes);

    int histSize[] = {30};
    float bins[] = {0,300};
    const float *ranges[] = {bins};

    for (int i=0; i<bgrPlanes.size(); i++)
        calcHist(&bgrPlanes[i], 1, 0, Mat(), this->histogram, 1, histSize, ranges, true, true);

    getHR(sqrt(image.rows * image.cols));
}

void Binarization::getHR(float sqrtHW){
    this->hr = 0;
    for (int i=0; i<this->histogram.rows; i++){
        if (this->histogram.at<float>(i,0) > sqrtHW){
            this->hr = (i * 10);
            break;
        }
    }
}

void Binarization::getCEI(){
    Mat cei = (this->grayscale - (this->hr + 50 * 0.4)) * 2;
    normalize(cei, this->cei, 0, 255, NORM_MINMAX, CV_32F);
    threshold(this->cei, this->ceiBin, 59, 255, THRESH_BINARY_INV);
}

void Binarization::getEdge(){
    float m1[] = {-1,0,1,-2,0,2,-1,0,1};
    float m2[] = {-2,-1,0,-1,0,1,0,1,2};
    float m3[] = {-1,-2,-1,0,0,0,1,2,1};
    float m4[] = {0,1,2,-1,0,1,-2,-1,0};

    Mat kernel1(3, 3, CV_32F, m1);
    Mat kernel2(3, 3, CV_32F, m2);
    Mat kernel3(3, 3, CV_32F, m3);
    Mat kernel4(3, 3, CV_32F, m4);

    Mat eg1, eg2, eg3, eg4;
    filter2D(this->grayscale, eg1, CV_32F, kernel1);
    eg1 = abs(eg1);

    filter2D(this->grayscale, eg2, CV_32F, kernel2);
    eg2 = abs(eg2);

    filter2D(this->grayscale, eg3, CV_32F, kernel3);
    eg3 = abs(eg3);

    filter2D(this->grayscale, eg4, CV_32F, kernel4);
    eg4 = abs(eg4);

    this->egAvg = scale((eg1 + eg2 + eg3 + eg4)/4);
    threshold(this->egAvg, this->egBin, 30, 255, THRESH_BINARY);
}

void Binarization::getTLI(){
    this->tli = Mat::ones(Size(this->grayscale.cols, this->grayscale.rows), CV_32F) * 255;
    this->tli -= this->egBin;
    this->tli -= this->ceiBin;
    threshold(this->tli, this->tli, 0, 255, THRESH_BINARY);

    Mat kernel = Mat::ones(Size(3, 3), CV_32F);
    erode(this->tli, this->tliErosion, kernel);
    threshold(this->tliErosion, this->tliErosion, 0, 255, THRESH_BINARY);
}

Mat Binarization::scale(Mat image){
    double min, max;
    minMaxLoc(image, &min, &max);

    Mat res = image / (max - min);
    minMaxLoc(res, &min, &max);
    res -= min;
    res *= 255;

    return res;
}
