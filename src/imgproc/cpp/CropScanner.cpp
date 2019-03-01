#include "CropScanner.hpp"

CropScanner::CropScanner() {};

bool compareContourAreas(vector<Point> contour1, vector<Point> contour2){
	double i = fabs(contourArea(Mat(contour1)));
	double j = fabs(contourArea(Mat(contour2)));
	return (i > j);
}

bool compareXCords(Point p1, Point p2){
	return (p1.x < p2.x);
}

bool compareYCords(Point p1, Point p2){
	return (p1.y < p2.y);
}

bool compareDistance(pair<Point, Point> p1, pair<Point, Point> p2){
	return (norm(p1.first - p1.second) < norm(p2.first - p2.second));
}

double distance(Point p1, Point p2){
	return sqrt(((p1.x - p2.x) * (p1.x - p2.x)) + ((p1.y - p2.y) * (p1.y - p2.y)));
}

void CropScanner::orderPoints(vector<Point> inpts, vector<Point> &ordered){
	sort(inpts.begin(), inpts.end(), compareXCords);
	vector<Point> lm(inpts.begin(), inpts.begin()+2);
	vector<Point> rm(inpts.end()-2, inpts.end());

	sort(lm.begin(), lm.end(), compareYCords);
	Point tl(lm[0]);
	Point bl(lm[1]);
	vector<pair<Point, Point> > tmp;

	for(size_t i = 0; i< rm.size(); i++){
		tmp.push_back(make_pair(tl, rm[i]));
	}

	sort(tmp.begin(), tmp.end(), compareDistance);
	Point tr(tmp[0].second);
	Point br(tmp[1].second);

	ordered.push_back(tl);
	ordered.push_back(tr);
	ordered.push_back(br);
	ordered.push_back(bl);
}

void CropScanner::fourPointTransform(Mat src, Mat &dst, vector<Point> pts){
	vector<Point> ordered_pts;
	orderPoints(pts, ordered_pts);

	double wa = distance(ordered_pts[2], ordered_pts[3]);
	double wb = distance(ordered_pts[1], ordered_pts[0]);
	double mw = max(wa, wb);

	double ha = distance(ordered_pts[1], ordered_pts[2]);
	double hb = distance(ordered_pts[0], ordered_pts[3]);
	double mh = max(ha, hb);

	Point2f src_[] ={
        Point2f(ordered_pts[0].x, ordered_pts[0].y),
        Point2f(ordered_pts[1].x, ordered_pts[1].y),
        Point2f(ordered_pts[2].x, ordered_pts[2].y),
        Point2f(ordered_pts[3].x, ordered_pts[3].y),
	};

	Point2f dst_[] ={
        Point2f(0,0),
        Point2f(mw-1, 0),
        Point2f(mw-1, mh-1),
        Point2f(0, mh-1)
	};

	Mat m = getPerspectiveTransform(src_, dst_);
	warpPerspective(src, dst, m, Size(mw, mh), BORDER_REPLICATE, INTER_LINEAR);
}

void CropScanner::get_edges(Mat input, Mat &output){
	Mat imageOpen, imageClosed, imageBlurred;
	Mat structuringElmt = getStructuringElement(MORPH_ELLIPSE, Size(1,1));

	morphologyEx(input, imageOpen, MORPH_OPEN, structuringElmt);
	morphologyEx(imageOpen, imageClosed, MORPH_CLOSE, structuringElmt);

	GaussianBlur(imageClosed, imageBlurred, Size(5,5), 0);
	Canny(imageBlurred, output, 75, 100);

    Mat kernel = getStructuringElement(MORPH_RECT, Size(9,9));
    dilate(output, output, kernel);
}

void CropScanner::process(Mat image, Mat &output, string data_base, string extension){
	Mat orig = image.clone();
    Mat pre_output = output.clone();

    double scale = 500.0;
    double ratio = image.rows / scale;

    Size s = Size(image.cols * (scale / double(image.rows)), scale);
	resize(image, image, s, INTER_AREA);

    Mat edged;
    get_edges(image, edged);

    vector<vector<Point>> contours, approx;
	vector<Vec4i> hierarchy;
	findContours(edged, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

    int area, mean_area, total_area = 0;
    vector<RotatedRect> min_rect(contours.size());

	for(int i=0; i<contours.size(); i++){
        min_rect[i] = minAreaRect(Mat(contours[i]));
        total_area += (min_rect[i].size.width * min_rect[i].size.height);
    }

    mean_area = (total_area/min_rect.size());
    vector<Point> points;

    for (int i=0; i<min_rect.size(); i++){
        area = (min_rect[i].size.width * min_rect[i].size.height);

        // rect area must be higher mean area from contoures and smaller total area
        // also smaller height and width
        if (area > mean_area * 0.1 && area < total_area * 0.3
            && min_rect[i].size.height < image.rows*0.5
            && min_rect[i].size.width < image.cols*0.9
        ){
            Point2f rect_points[4]; 
            min_rect[i].points(rect_points);

            for (int j=0; j<4; j++){
                points.push_back(Point(rect_points[j].x*ratio, rect_points[j].y*ratio));
                line(pre_output, rect_points[j]*ratio, rect_points[(j+1)%4]*ratio, Scalar(0, 255, 0), 2, 8);
            }
        }
    }

    min_rect.clear();
    min_rect.push_back(minAreaRect(Mat(points)));

    for (int i=0; i<min_rect.size(); i++){
        vector<Point> p;
        Point2f rect_points[4];
        min_rect[i].points(rect_points);

        for(int j=0; j<4; j++){
            p.push_back(Point(rect_points[j].x, rect_points[j].y));
            line(pre_output, rect_points[j], rect_points[(j+1)%4], Scalar(0, 255, 0), 2, 8);
        }
        approx.push_back(p);
    }
    // imwrite(data_base + "_2_crop_detection" + extension, pre_output);

    fourPointTransform(orig, output, approx[0]);
    
    int pad = output.cols * 0.25;
    copyMakeBorder(output, output, 0, 0, pad, pad, BORDER_CONSTANT, 255);
}
