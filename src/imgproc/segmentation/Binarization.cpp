#include "Binarization.hpp"

Binarization::Binarization() {};

Mat Binarization::otsu(Mat image) {
    Mat grayscale, smoothed_img, binary_img;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    // Noise reduction (Currently a basic filter).
    blur(grayscale, smoothed_img, Size(3, 3), Point(-1, -1));

    // OTSU threshold and Binarization.
    threshold(smoothed_img, binary_img, 0.0, 255, THRESH_BINARY | THRESH_OTSU);

    return binary_img;
}

Mat Binarization::illumination_sauvola(Mat image) {
    Mat grayscale;
    cvtColor(image, grayscale, COLOR_BGR2GRAY);

    int width = grayscale.size().width;
    int height = grayscale.size().height;
    float sqrt_hw = sqrt(height * width);

    float bins[] = {0,300};

    this->get_hist(image, bins);

    this->hr = get_hr(sqrt_hw);
    this->cei = get_cei(grayscale, this->hr, 0.4);

    edge_detection(grayscale);

    return this->cei;
}

void Binarization::edge_detection(Mat grayscale){

    

}

void Binarization::get_hist(Mat image, float bins[]){
    vector<Mat> bgr_planes;
    split(image, bgr_planes);

    int histSize[] = {30};
    const float *ranges[] = {bins};

    for (int i=0; i<bgr_planes.size(); i++){
        calcHist(&bgr_planes[i], 1, 0, Mat(), this->histogram, 1, histSize, ranges, true, true);
    }

    int end = bins[1];
    int step = (int)bins[1]/histSize[0];

    for (int i=0; i<end; i+=step){
        this->bins.push_back(i);
    }
}

float Binarization::get_hr(float sqrt_hw){
    for (int i=0; i<this->histogram.rows; i++){
        if (this->histogram.at<float>(i, 0) > sqrt_hw)
            return i * 10;
    }
    return 0;
}

Mat Binarization::get_cei(Mat grayscale, float hr, float c){
    Mat cei_norm, cei = (grayscale - (hr + 50 * c)) * 2;
    normalize(cei, cei_norm, 0, 255, NORM_MINMAX);
    return cei_norm;
}