#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

class Scanner {
    public:
        Scanner();
        void process(Mat image, Mat &output);
	    bool cropped;

    private:
        void processEdge(Mat input, Mat &output, int openKSize, int closeKSize, bool gaussianBlur);
        void fourPointTransform(Mat src, Mat &dst, vector<Point> pts);
        void orderPoints(vector<Point> inpts, vector<Point> &ordered);
};
