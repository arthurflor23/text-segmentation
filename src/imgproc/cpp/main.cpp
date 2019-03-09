#include "Binarization.hpp"
#include "Scanner.hpp"
#include "LineSegmentation.hpp"
#include "WordSegmentation.hpp"
#include <opencv2/core/utils/filesystem.hpp>

using namespace cv::utils::fs;

int main(int argc, char *argv[]) {

    string srcPath = argv[1], outPath = argv[2];
    string name = argv[3], extension = argv[4];
    string srcBase = (outPath + name);

    int illumination = stoi(argv[5]);
    int thresholdMethod = stoi(argv[6]);

    string linesPath = join(outPath, "lines");
    string wordsPath = join(outPath, "words");

    Mat image = imread(srcPath);
    createDirectory(outPath);
    imwrite(srcBase + extension, image);


    // START Step 1: crop //
    Scanner *scanner = new Scanner();
    scanner->process(image, image);
    imwrite(srcBase + "_1_crop" + extension, image);
    // END Step 1 //


    // START Step 1.1: resize and definitions //
    int newW = 1024;
    int newH = ((newW * image.rows) / image.cols);

    int chunksNumber = 8;
    int chunksProcess = 4;

    resize(image, image, Size(newW, newH));
    // END Step 1.1 //


    // START Step 2: binarization //
    Binarization *threshold = new Binarization();
    threshold->binarize(image, image, illumination, thresholdMethod);
    imwrite(srcBase + "_2_binary" + extension, image);
    // END Step 2 //


    // START Step 3: line segmentation //
    createDirectory(linesPath);
    LineSegmentation *line = new LineSegmentation(srcBase, extension);
    vector<Mat> lines;
    line->segment(image, lines, chunksNumber, chunksProcess);
    // END Step 3 //


    // START Step 4: word segmentation //
    createDirectory(wordsPath);
    WordSegmentation *word = new WordSegmentation(srcBase, extension);
    word->setKernel(11, 11, 7);

    for (int i=0; i<lines.size(); i++) {
        string lineNumber = "line_" + to_string((i+1)*1e-6).substr(5);
        imwrite(join(linesPath,  lineNumber + extension), lines[i]);

        string wordPath = join(wordsPath, lineNumber);
        createDirectory(wordPath);

        vector<Mat> words;
        word->segment(lines[i], words);
        imwrite(join(wordsPath, lineNumber + "_summary" + extension), words[0]);

        for (int j=1; j<words.size(); j++) {
            string wordNumber = "word_" + to_string((j)*1e-6).substr(5);
            imwrite(join(wordPath, wordNumber + extension), words[j]);
        }
    }
    // END Step 4 //


    return 0;
}