#include "Binarization.hpp"
#include "LineSegmentation.hpp"
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv::utils::fs;

int main(int argc, char *argv[]) {

    string src_path = argv[1], out_path = argv[2];
    string name = argv[3], extension = argv[4];
    string src_base = (out_path + name);

    bool illumination = argv[5];
    int threshold_method = stoi(argv[6]);

    string lines_path = join(out_path, "lines");
    string words_path = join(out_path, "words");

    Mat image = imread(src_path);

    createDirectory(out_path);
    imwrite(src_base + extension, image);

    // crop
    // ....
    

    // binarization
    Binarization *threshold = new Binarization();    
    threshold->binarize(image, illumination, threshold_method);

    imwrite(src_base + "_binary" + extension, threshold->binary);

    // line segmentation
    LineSegmentation *line = new LineSegmentation(threshold->binary);

    delete threshold;
    vector<cv::Mat> lines = line->segment(src_base, extension);
    
    createDirectory(lines_path);
    for (int i=0; i< lines.size(); i++) {
        string number = to_string((i+1)*1e-6).substr(5);
        imwrite(join(lines_path, "line_" + number + extension), lines[i]);
    }


    // word segmentation
    // createDirectory(words_path);


    return 0;
}