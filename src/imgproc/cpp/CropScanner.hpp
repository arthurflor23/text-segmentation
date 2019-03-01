#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

class CropScanner {
    public:
        CropScanner();
        void process(Mat image, Mat &output, string data_base, string extension);

    private:
        void get_rect_text(Mat binary, Rect &output);
        void get_edges(Mat input, Mat &output);

        void fourPointTransform(Mat src, Mat &dst, vector<Point> pts);
        void orderPoints(vector<Point> inpts, vector<Point> &ordered);
};
