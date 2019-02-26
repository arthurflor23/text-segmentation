#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

class Binarization {
    public:
        Binarization();
        Mat otsu(Mat grayscale);
        Mat illumination_sauvola(Mat img);

        Mat cei;
        Mat histogram;
        vector<int> bins;

        int width;
        int height;

        float sqrt_hw;
        float hr;

    private:
        void get_hist(Mat image, float bins[]);
        float get_hr(float sqrt_hw);
        Mat get_cei(Mat grayscale, float hr, float c);
        void edge_detection(Mat grayscale);
};
