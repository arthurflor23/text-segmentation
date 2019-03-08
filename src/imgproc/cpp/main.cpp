#include "Binarization.hpp"
#include "Scanner.hpp"
#include "LineSegmentation.hpp"
#include "WordSegmentation.hpp"
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
    scanner->process(image, image);
    imwrite(src_base + "_1_crop" + extension, image);
    // END Step 1 //


    // START Step 1.1: resize and definitions //
    int new_w = 1024;
    int new_h = ((new_w * image.rows) / image.cols);

    int chunks_number = 8;
    int chunks_process = 4;

    resize(image, image, Size(new_w, new_h));
    // END Step 1.1 //


    // START Step 2: binarization //
    Binarization *threshold = new Binarization();
    threshold->binarize(image, image, illumination, threshold_method);
    imwrite(src_base + "_2_binary" + extension, image);
    // END Step 2 //


    // START Step 3: line segmentation //
    createDirectory(lines_path);
    LineSegmentation *line = new LineSegmentation(src_base, extension);
    vector<Mat> lines;
    line->segment(image, lines, chunks_number, chunks_process);
    // END Step 3 //


    // START Step 4: word segmentation //
    createDirectory(words_path);
    WordSegmentation *word = new WordSegmentation(src_base, extension);
    word->set_kernel(11, 11, 7);

    for (int i=0; i<lines.size(); i++) {
        string l_number = "line_" + to_string((i+1)*1e-6).substr(5);
        imwrite(join(lines_path,  l_number + extension), lines[i]);

        string word_path = join(words_path, l_number);
        createDirectory(word_path);

        vector<Mat> words;
        word->segment(lines[i], words);
        imwrite(join(words_path, l_number + "_summary" + extension), words[0]);

        for (int j=1; j<words.size(); j++) {
            string w_number = "word_" + to_string((j)*1e-6).substr(5);
            imwrite(join(word_path, w_number + extension), words[j]);
        }
    }
    // END Step 4 //


    return 0;
}