#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

class WordSegmentation {
    public:
        WordSegmentation(string srcBase, string extension);

        void segment(Mat line, vector<Mat> &words);
        void setKernel(int kernelSize, int sigma, int theta);

    private:
        string srcBase; 
        string extension;
        Mat kernel;
};
