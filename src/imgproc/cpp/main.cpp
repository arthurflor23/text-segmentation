#include "Binarization.hpp"
#include "Scanner.hpp"
#include "LineSegmentation.hpp"
#include "WordSegmentation.hpp"
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv::utils::fs;

int main(int argc, char *argv[]) {

    string srcPath = argv[1];
    string outPath = argv[2];

    Mat image = imread(srcPath);

    String name = outPath.substr(outPath.find_last_of("/\\") + 1);
    name = name.substr(0, name.find("."));

    string extension = ".png";
    string wordsPath = join(outPath, "words");

    // START Step 1: crop //
    Scanner *scanner = new Scanner();
    Mat imageCropped;
    scanner->process(image, imageCropped);
    // END Step 1 //


    // START Step 1.1: resize and definitions //
    int newW = 1280;
    int newH = ((newW * imageCropped.rows) / imageCropped.cols);

    if (imageCropped.cols > newW)
        resize(imageCropped, imageCropped, Size(newW, newH));

    int chunksNumber = 8;
    int chunksProcess = 4;
    // END Step 1.1 //


    // START Step 2: binarization //
    Binarization *threshold = new Binarization();
    Mat imageBinary;
    threshold->binarize(imageCropped, imageBinary, 1); // niblack = 0 | sauvola = 1 | wolf = 2 | otsu = 3
    // END Step 2 //


    // START Step 3: line segmentation //
    LineSegmentation *line = new LineSegmentation();
    vector<Mat> lines;
    Mat imageLines = imageBinary.clone();
    line->segment(imageLines, lines, chunksNumber, chunksProcess);
    // END Step 3 //


    // START Step 4: word segmentation //
    WordSegmentation *word = new WordSegmentation();
    vector<Mat> summary;
    word->setKernel(11, 11, 7);

    for (int i=0; i<lines.size(); i++) {
        string lineIndex = to_string((i+1)*1e-6).substr(5);

        vector<Mat> words;
        word->segment(lines[i], words);

        summary.push_back(words[0]);
        words.erase(words.begin());

        createDirectories(wordsPath);

        for (int j=0; j<words.size(); j++) {
            string wordIndex = lineIndex + "_" + to_string((j+1)*1e-6).substr(5);
            imwrite(join(wordsPath, wordIndex + extension), words[j]);
        }
    }
    // END Step 4 //


    createDirectories(outPath);
    // imwrite(join(outPath, name + extension), image);
    imwrite(join(outPath, name + "_1_crop" + extension), imageCropped);
    imwrite(join(outPath, name + "_2_binary" + extension), imageBinary);
    imwrite(join(outPath, name + "_3_lines" + extension), imageLines);

    for (int i=0; i<summary.size(); i++){
        string index = "_4_summary_" + to_string((i+1)*1e-6).substr(5);
        imwrite(join(outPath, name + index + extension), summary[i]);
    }


    return 0;
}
