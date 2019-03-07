#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

class WordSegmentation {
    public:
        WordSegmentation(string src_base, string extension);

        void segment(Mat line, vector<Mat> &words);
        void set_kernel(int kernel_size, int sigma, int theta, int min_area);

    private:
        string src_base; 
        string extension;

        Mat kernel;
        int min_area;
};
