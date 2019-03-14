#include <iostream>
#include <map>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

typedef int valleyID;

using namespace cv;
using namespace std;

class LineSegmentation;
class Region;
class Valley;

class Line {
    public:
        Line(int initialValleyID);
        friend class LineSegmentation;
        friend class Region;

    private:
        Region *above;
        Region *below;
        vector<valleyID> valleysID;
        int minRowPosition;
        int maxRowPosition;
        vector<Point> points;

        void generateInitialPoints(int chunksNumber, int chunkWidth, int imgWidth, map<int, Valley *> mapValley);
        static bool compMinRowPosition(const Line *a, const Line *b);
};

class Peak {
    public:
        Peak() {}
        Peak(int p, int v): position(p), value(v){}
        Peak(int p, int v, int s, int e): position(p), value(v){}

        int position;
        int value;

        bool operator<(const Peak &p) const;
        static bool comp(const Peak &a, const Peak &b);
};

class Valley {
    public:
        Valley(): valleyID(ID++), used(false){}
        Valley(int cID, int p): chunkIndex(cID), valleyID(ID++), position(p), used(false){}

        static int ID;
        int chunkIndex;
        int valleyID;
        int position;
        bool used;
        Line *line;

        static bool comp(const Valley *a, const Valley *b);
};

class Region {
    public:
        Region(Line *top, Line *bottom);
        friend class LineSegmentation;

    private:
        int regionID;
        Mat region;
        Line *top;
        Line *bottom;
        int height;
        int rowOffset;
        Mat covariance;
        Vec2f mean;

        bool updateRegion(Mat &img, int);
        void calculateMean();
        void calculateCovariance();
        double biVariateGaussianDensity(Mat point);
};

class Chunk {
    public:
        Chunk(int o, int c, int w, Mat i);
        friend class LineSegmentation;
        
        int findPeaksValleys(map<int, Valley *> &mapValley);

    private:
        int index;
        int startCol;
        int width;
        Mat img;
        vector<int> histogram;
        vector<Peak> peaks;
        vector<Valley *> valleys;
        int avgHeight;
        int avgWhiteHeight;
        int linesCount;

        void calculateHistogram();
};

class LineSegmentation {
    public:
        LineSegmentation();

        Mat binaryImg;
        vector<Rect> contours;
        Mat contoursDrawing;
        Mat linesDrawing;

        void segment(Mat &input, vector<Mat> &output, int chunksNumber, int chunksProcess);
        void getContours();
        void generateChunks();
        void getInitialLines();
        void getRegions(vector<Mat> &output);

        void generateRegions();
        void repairLines();
        void deslant(Mat image, Mat &output, int bgcolor);

    private:
        string srcBase;
        string extension;

        int chunksNumber;
        int chunksToProcess;

        bool notPrimesArr[100007];
        vector<int> primes;

        int chunkWidth;
        vector<Chunk *> chunks;
        map<int, Valley *> mapValley;
        vector<Line *> initialLines;
        vector<Region *> lineRegions;
        int avgLineHeight;
        int predictedLineHeight;

        void sieve();
        void addPrimesToVector(int, vector<int> &);
        void printLines(Mat &inputOutput);

        Line * connectValleys(int i, Valley *currentValley, Line *line, int valleysMinAbsDist);
        bool componentBelongsToAboveRegion(Line &, Rect &);
};
