#include <iostream>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

class Binarization {
    public:
        Binarization();
        void binarize(Mat image, Mat &output, int illumination, int option);

        Mat grayscale;
        Mat histogram;
        float hr;

        Mat cei;
        Mat cei_bin;
        Mat eg_avg;
        Mat eg_bin;
        Mat tli;
        Mat tli_erosion;
        Mat ldi;

    private:
        void get_histogram(Mat image);
        void get_hr(float sqrt_hw);
        void get_cei();
        void get_edge();
        void get_tli();

        void light_distribution();
        void otsu(Mat grayscale, Mat &output);
        void local_thresholding(Mat im, Mat &output, int option, int winx, int winy, double k, double dR);

        double calc_local_stats(Mat &im, Mat &map_m, Mat &map_s, int winx, int winy);
        Mat scale(Mat image);
};
