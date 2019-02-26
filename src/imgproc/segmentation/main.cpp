#include "Binarization.hpp"
#include "LineSegmentation.hpp"
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv::utils::fs;

int main(int argc, char *argv[]) {

    string data_path = argv[1];
    string out_path = argv[2];
    string name = argv[3];
    string extension = argv[4];

    string data_base = (out_path + name);
    string lines_path = join(out_path, "lines");
    string words_path = join(out_path, "words");

    createDirectory(out_path);

    Mat image = imread(data_path);
    imwrite(data_base + extension, image);

    // crop
    // ....
    

    
    // binarization
    Binarization *threshold = new Binarization();
    
    // Mat binary_image = threshold->otsu(image);
    Mat binary_image = threshold->illumination_sauvola(image);



    imwrite(data_base + "_binary" + extension, binary_image);

    // line segmentation
    createDirectory(lines_path);

    // LineSegmentation *line = new LineSegmentation(binary_image);
    // vector<cv::Mat> lines = line->segment(data_base, extension);

    // for (int i=0; i< lines.size(); i++) {
    //     string number = to_string((i+1)*1e-6).substr(5);
    //     imwrite(join(lines_path, "line_" + number + extension), lines[i]);
    // }

    // word segmentation
    createDirectory(words_path);

    
    return 0;
}