#include "WordSegmentation.hpp"

WordSegmentation::WordSegmentation(string src_base, string extension) {
    this->src_base = src_base;
    this->extension = extension;
};

bool compare_cords(const Rect &p1, const Rect &p2){
	return (p1.area() > 10) && (p2.area() > 10) && (p1.x < p2.x || p1.y < p2.y);
}

bool compare_x_cord(const Rect &p1, const Rect &p2){
	return (p1.x < p2.x);
}

void WordSegmentation::segment(Mat line, vector<Mat> &words){
    copyMakeBorder(line, line, 10, 10, 10, 10, BORDER_CONSTANT, 255);

    Mat img_filtered;
    filter2D(line, img_filtered, CV_8UC1, this->kernel);
    threshold(img_filtered, img_filtered, 0, 255, THRESH_BINARY | THRESH_OTSU);

	vector<vector<Point>> contours;
   	vector<Vec4i> hierarchy;

    findContours(img_filtered, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
    Mat edged = Mat::zeros(Size(line.cols, line.rows), CV_8UC1);
    
    for (int i=0; i<contours.size(); i++){
        Rect r = boundingRect(Mat(contours[i]));
        if (r.area() < line.rows*line.cols*0.9)
            rectangle(edged, r.tl(), r.br(), 255, 2, 8, 0);
    }
    findContours(edged, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    
    edged = Mat::zeros(Size(line.cols, line.rows), CV_8UC1);
    vector<Rect> bound_rect;

    for (int i=0; i<contours.size(); i++)
        bound_rect.push_back(boundingRect(Mat(contours[i])));
    sort(bound_rect.begin(), bound_rect.end(), compare_x_cord);

    Mat image_color;
    cvtColor(line, image_color, COLOR_GRAY2BGR);

    for (int i=0; i<bound_rect.size(); i++){
        if (i < bound_rect.size()-1){
            if (bound_rect[i+1].tl().x >= bound_rect[i].tl().x && 
                bound_rect[i+1].br().x <= bound_rect[i].br().x
            ){
                int min_x = min(bound_rect[i].tl().x, bound_rect[i+1].tl().x);
                int min_y = min(bound_rect[i].tl().y, bound_rect[i+1].tl().y);
                int max_y = max(bound_rect[i].br().y, bound_rect[i+1].br().y);

                int width = max(bound_rect[i].width, bound_rect[i+1].width);
                int height = abs(min_y - max_y);

                bound_rect[i+1] = Rect(min_x, min_y, width, height);
                i++;
            }
        }
    }
    sort(bound_rect.begin(), bound_rect.end(), compare_cords);

    for (int i=0; i<bound_rect.size(); i++){
        Mat cropped;
        line(bound_rect[i]).copyTo(cropped);

        rectangle(image_color, bound_rect[i].tl(), bound_rect[i].br(), Vec3b(0,0,255), 2, 8, 0);
        words.push_back(cropped);
    }

    words.push_back(image_color);
    rotate(words.rbegin(), words.rbegin()+1, words.rend());
}

void WordSegmentation::set_kernel(int kernel_size, int sigma, int theta){
    Mat kernel = Mat::zeros(Size(kernel_size, kernel_size), CV_32F);
    float sigma_x = sigma;
	float sigma_y = sigma * theta;

    for (int i=0; i<kernel_size; i++){
        for (int j=0; j<kernel_size; j++){
            float x = i - (kernel_size / 2);
            float y = j - (kernel_size / 2);

            float exp_term = exp((-pow(x,2) / (2*sigma_x)) - (pow(y,2) / (2*sigma_y)));

            float x_term = (pow(x,2) - pow(sigma_x,2)) / (2 * CV_PI * pow(sigma_x,5) * sigma_y);
            float y_term = (pow(y,2) - pow(sigma_y,2)) / (2 * CV_PI * pow(sigma_y,5) * sigma_x);

            kernel.at<float>(i,j) = (x_term + y_term) * exp_term;
        }
    }

    this->kernel = kernel / sum(kernel)[0];
}
