#include <iostream>
#include <map>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define CHUNKS_NUMBER 20
#define CHUNKS_TO_BE_PROCESSED 5

typedef int valley_id;

using namespace cv;
using namespace std;

class LineSegmentation;
class Region;
class Valley;

class Line {
    public:
        Line(int initial_valley_id);
        friend class LineSegmentation;
        friend class Region;

    private:
        Region *above;
        Region *below;
        vector<valley_id> valleys_ids;
        int min_row_position;
        int max_row_position;
        vector<Point> points;

        void generate_initial_points(int chunk_width, int img_width, map<int, Valley *> map_valley);
        static bool comp_min_row_position(const Line *a, const Line *b);
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
        Valley(): valley_id(ID++), used(false){}
        Valley(int c_id, int p): chunk_index(c_id), valley_id(ID++), position(p), used(false){}

        static int ID;
        int chunk_index;
        int valley_id;
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
        int region_id;
        Mat region;
        Line *top;
        Line *bottom;
        int height;
        int row_offset;
        Mat covariance;
        Vec2f mean;

        bool update_region(Mat &img, int);
        void calculate_mean();
        void calculate_covariance();
        double bi_variate_gaussian_density(Mat point);
};

class Chunk {
    public:
        Chunk(int o, int c, int w, Mat i);
        friend class LineSegmentation;
        
        int find_peaks_valleys(map<int, Valley *> &map_valley);

    private:
        int index;
        int start_col;
        int width;
        Mat img;
        vector<int> histogram;
        vector<Peak> peaks;
        vector<Valley *> valleys;
        int avg_height;
        int avg_white_height;
        int lines_count;

        void calculate_histogram();
};

class LineSegmentation {
    public:
        LineSegmentation();

        Mat binary_img;
        vector<Rect> contours;
        Mat contours_drawing;
        Mat lines_drawing;

        void segment(Mat input, vector<Mat> &output, string data_base, string extension);
        void find_contours();
        void generate_chunks();
        void get_initial_lines();

        void generate_regions();
        void repair_lines();
        void get_regions(vector<Mat> &output);
        void generate_image_with_lines();

    private:
        bool not_primes_arr[100007];
        vector<int> primes;

        int chunk_width;
        vector<Chunk *> chunks;
        map<int, Chunk *> chunk_map;
        map<int, Valley *> map_valley;
        vector<Line *> initial_lines;
        vector<Region *> line_regions;
        int avg_line_height;
        int predicted_line_height;

        void sieve();
        void add_primes_to_vector(int, vector<int> &);

        Line * connect_valleys(int i, Valley *current_valley, Line *line, int valleys_min_abs_dist);
        bool component_belongs_to_above_region(Line &, Rect &);
};
