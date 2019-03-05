#include "Binarization.hpp"
#include "Scanner.hpp"
#include "LineSegmentation.hpp"
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv::utils::fs;

int main(int argc, char *argv[]) {

    string src_path = argv[1], out_path = argv[2];
    string name = argv[3], extension = argv[4];
    string src_base = (out_path + name);

    int illumination = stoi(argv[5]);
    int threshold_method = stoi(argv[6]);

    string lines_path = join(out_path, "lines");
    string words_path = join(out_path, "words");

    Mat image = imread(src_path);
    createDirectory(out_path);
    imwrite(src_base + extension, image);


    // START Step 1: crop //
    Scanner *scanner = new Scanner();
    scanner->process(image, image, src_base, extension);
    imwrite(src_base + "_1_crop" + extension, image);
    // END Step 1 //


    // START Step 1.1: resize and definitions //
    int new_w = 1024;
    int new_h = ((new_w * image.rows) / image.cols);
    int chunks_number = 20, chunks_process = 5;
    
    if (image.cols < new_w)
        chunks_number = 12;

    resize(image, image, Size(new_w, new_h));
    // END Step 1.1 //


    // START Step 2: binarization //
    Binarization *threshold = new Binarization();
    threshold->binarize(image, image, illumination, threshold_method);
    imwrite(src_base + "_2_binary" + extension, image);
    // END Step 2 //


    // START Step 3: line segmentation //
    LineSegmentation *line = new LineSegmentation(chunks_number, chunks_process);
    vector<cv::Mat> lines;
    line->segment(image, lines, src_base, extension);
    
    createDirectory(lines_path);
    for (int i=0; i< lines.size(); i++) {
        string number = to_string((i+1)*1e-6).substr(5);
        imwrite(join(lines_path, "line_" + number + extension), lines[i]);
    }
    // END Step 3 //


    // START Step 4: word segmentation //
    // createDirectory(words_path);

    // END Step 4 //


    return 0;
}