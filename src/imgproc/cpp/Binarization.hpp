#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

class Binarization {
    public:
        Binarization();
        void binarize(Mat image, Mat &output, int option);

        Mat grayscale;
        Mat histogram;
        float hr;

        Mat cei;
        Mat ceiBin;
        Mat egAvg;
        Mat egBin;
        Mat tli;
        Mat tliErosion;
        Mat ldi;

    private:
        void getHistogram(Mat image);
        void getHR(float sqrtHW);
        void getCEI();
        void getEdge();
        void getTLI();

        void lightDistribution();
        void thresholdImg(Mat im, Mat &output, int option, int winx, int winy, double k, double dR);

        double calcLocalStats(Mat &im, Mat &mapM, Mat &mapS, int winx, int winy);
        Mat scale(Mat image);
};
