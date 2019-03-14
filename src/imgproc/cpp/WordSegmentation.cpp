#include "WordSegmentation.hpp"

WordSegmentation::WordSegmentation() {};

bool compareCords(const Rect &p1, const Rect &p2){
	// return (p1.area() > 10) && (p2.area() > 10) && (p1.x < p2.x);
	return (p1.x < p2.x);
}

void WordSegmentation::printContours(Mat image, vector<vector<Point>> contours, vector<Vec4i> hierarchy, int idx){
    for(int i = idx; i >= 0; i = hierarchy[i][0]){
        drawContours(image, contours, i, Scalar(255));
        for(int j=hierarchy[i][2]; j>=0; j=hierarchy[j][0])
            printContours(image, contours, hierarchy, hierarchy[j][2]);
    }
}

void WordSegmentation::processBounds(Mat &image, vector<Rect> &boundRect){
    vector<vector<Point>> contours;
   	vector<Vec4i> hierarchy;
    Mat edged;
    int lastNumber = 0;

    while(true){
        findContours(image, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        edged = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);

        for (int i=0; i<contours.size(); i++){
            Rect r = boundingRect(Mat(contours[i]));
            rectangle(edged, r.tl(), r.br(), 255, 2, 8, 0);
        }
        
        printContours(edged, contours, hierarchy, 0);
        image = edged;

        if (contours.size() == lastNumber) break;
        lastNumber = contours.size();
    }

    for (int i=0; i<contours.size(); i++)
        boundRect.push_back(boundingRect(Mat(contours[i])));
    sort(boundRect.begin(), boundRect.end(), compareCords);

    int i=0;
    while (i<boundRect.size()-1){
        if (boundRect[i].tl().x <= boundRect[i+1].tl().x && 
            boundRect[i].br().x >= boundRect[i+1].br().x
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
}

void WordSegmentation::segment(Mat line, vector<Mat> &words){
    copyMakeBorder(line, line, 10, 10, 10, 10, BORDER_CONSTANT, 255);

    Mat imgFiltered;
    filter2D(line, imgFiltered, CV_8UC1, this->kernel);
    threshold(imgFiltered, imgFiltered, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    vector<Rect> boundRect;
    processBounds(imgFiltered, boundRect);

    Mat imageColor;
    cvtColor(line, imageColor, COLOR_GRAY2BGR);

    for (int i=0; i<boundRect.size(); i++){
        Mat cropped;
        line(boundRect[i]).copyTo(cropped);

        rectangle(imageColor, boundRect[i].tl(), boundRect[i].br(), Vec3b(0,0,255), 2, 8, 0);
        putText(imageColor, to_string(i+1), boundRect[i].tl(), FONT_HERSHEY_SIMPLEX, 0.5, Vec3b(255,0,0), 2);
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
