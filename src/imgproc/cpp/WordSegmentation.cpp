#include "WordSegmentation.hpp"

WordSegmentation::WordSegmentation(string srcBase, string extension) {
    this->srcBase = srcBase;
    this->extension = extension;
};

bool compareCords(const Rect &p1, const Rect &p2){
	return (p1.area() > 10) && (p2.area() > 10) && (p1.x < p2.x || p1.y < p2.y);
}

bool compareXCords(const Rect &p1, const Rect &p2){
	return (p1.x < p2.x);
}

void WordSegmentation::segment(Mat line, vector<Mat> &words){
    copyMakeBorder(line, line, 10, 10, 10, 10, BORDER_CONSTANT, 255);

    Mat imgFiltered;
    filter2D(line, imgFiltered, CV_8UC1, this->kernel);
    threshold(imgFiltered, imgFiltered, 0, 255, THRESH_BINARY | THRESH_OTSU);

	vector<vector<Point>> contours;
   	vector<Vec4i> hierarchy;

    findContours(imgFiltered, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
    Mat edged = Mat::zeros(Size(line.cols, line.rows), CV_8UC1);
    
    for (int i=0; i<contours.size(); i++){
        Rect r = boundingRect(Mat(contours[i]));
        if (r.area() < line.rows*line.cols*0.9)
            rectangle(edged, r.tl(), r.br(), 255, 2, 8, 0);
    }
    findContours(edged, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    edged = Mat::zeros(Size(line.cols, line.rows), CV_8UC1);
    vector<Rect> boundRect;

    for (int i=0; i<contours.size(); i++)
        boundRect.push_back(boundingRect(Mat(contours[i])));
    sort(boundRect.begin(), boundRect.end(), compareXCords);

    Mat imageColor;
    cvtColor(line, imageColor, COLOR_GRAY2BGR);

    int i=0;
    while (i<boundRect.size()-1){
        if (boundRect[i+1].tl().x >= boundRect[i].tl().x && 
            boundRect[i+1].br().x <= boundRect[i].br().x
        ){
            int minX = min(boundRect[i].tl().x, boundRect[i+1].tl().x);
            int minY = min(boundRect[i].tl().y, boundRect[i+1].tl().y);
            int maxY = max(boundRect[i].br().y, boundRect[i+1].br().y);

            int width = max(boundRect[i].width, boundRect[i+1].width);
            int height = abs(minY - maxY);

            boundRect[i+1] = Rect(minX, minY, width, height);
            boundRect.erase(boundRect.begin() + i);
            continue;
        }
        ++i;
    }
    sort(boundRect.begin(), boundRect.end(), compareCords);

    for (int i=0; i<boundRect.size(); i++){
        Mat cropped;
        line(boundRect[i]).copyTo(cropped);

        rectangle(imageColor, boundRect[i].tl(), boundRect[i].br(), Vec3b(0,0,255), 2, 8, 0);
        words.push_back(cropped);
    }

    words.push_back(imageColor);
    rotate(words.rbegin(), words.rbegin()+1, words.rend());
}

void WordSegmentation::setKernel(int kernelSize, int sigma, int theta){
    Mat kernel = Mat::zeros(Size(kernelSize, kernelSize), CV_32F);
    float sigmaX = sigma;
	float sigmaY = sigma * theta;

    for (int i=0; i<kernelSize; i++){
        for (int j=0; j<kernelSize; j++){
            float x = i - (kernelSize / 2);
            float y = j - (kernelSize / 2);

            float termExp = exp((-pow(x,2) / (2*sigmaX)) - (pow(y,2) / (2*sigmaY)));

            float termX = (pow(x,2) - pow(sigmaX,2)) / (2 * CV_PI * pow(sigmaX,5) * sigmaY);
            float termY = (pow(y,2) - pow(sigmaY,2)) / (2 * CV_PI * pow(sigmaY,5) * sigmaX);

            kernel.at<float>(i,j) = (termX + termY) * termExp;
        }
    }

    this->kernel = kernel / sum(kernel)[0];
}
