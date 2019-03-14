#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

class WordSegmentation {
    public:
        WordSegmentation();

        void segment(Mat line, vector<Mat> &words);
        void setKernel(int kernelSize, int sigma, int theta);

    private:
        string srcBase; 
        string extension;
        Mat kernel;

        void printContours(Mat image, vector<vector<Point>> contours, vector<Vec4i> hierarchy, int idx);
        void processBounds(Mat &image, vector<Rect> &boundRect);
};
