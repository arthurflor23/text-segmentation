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
			Point2f(mw - 1, 0),
			Point2f(mw - 1, mh - 1),
			Point2f(0, mh - 1)
	};

	Mat m = getPerspectiveTransform(src_, dst_);
	warpPerspective(src, dst, m, Size(mw, mh));
}

void CropScanner::get_edges(Mat input, Mat &output){
	Mat imageOpen, imageClosed, imageBlurred;
	Mat structuringElmt = getStructuringElement(MORPH_ELLIPSE, Size(4,4));

	morphologyEx(input, imageOpen, MORPH_OPEN, structuringElmt);
	morphologyEx(imageOpen, imageClosed, MORPH_CLOSE, structuringElmt);

	GaussianBlur(imageClosed, imageBlurred, Size(7, 7), 0);
	Canny(imageBlurred, output, 75, 100);
}

void CropScanner::process(Mat image, Mat &output){
	Mat orig = image.clone();
    double scale = 500.0;
    double ratio = image.rows / scale;

    Size s = Size(image.cols * (scale / double(image.rows)), scale);
	resize(image, image, s, INTER_AREA);

    Mat edged;
    get_edges(image, edged);

    vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(edged, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

    // vector<vector<Point>> contours_poly(contours.size());
    // vector<Rect> bound_rect(contours.size()-1);

    // for (int i = 0; i < contours.size() - 1; i++) {
    //     approxPolyDP(Mat(contours[i]), contours_poly[i], 1, true);
    //     bound_rect[i] = boundingRect(Mat(contours_poly[i]));
    // }

    // Rect2d rectangle3;
    // vector<Rect> merged_rectangles;
    // bool is_repeated;

    // int min_x = image.cols, max_x = 0;
    // int min_y = image.rows, max_y = 0;
    // int total_area = 0, mean_area = 0;
    // int best_area = 0;

    // for (int i = 0; i < bound_rect.size(); i++) {
    //     is_repeated = false;

    //     for (int j = i + 1; j < bound_rect.size(); j++) {
    //         rectangle3 = bound_rect[i] & bound_rect[j];

    //         if ((rectangle3.area() == bound_rect[i].area()) || (rectangle3.area() == bound_rect[j].area())) {
    //             is_repeated = true;
    //             rectangle3 = bound_rect[i] | bound_rect[j];
    //             Rect2d merged_rectangle(rectangle3.tl().x, rectangle3.tl().y, rectangle3.width, rectangle3.height);

    //             // if (j == bound_rect.size() - 2)
    //                 // merged_rectangles.push_back(merged_rectangle);
    //                 // total_area += merged_rectangle.area(); 

    //             bound_rect[j] = merged_rectangle;
    //         }
    //     }
    //     if (!is_repeated)
    //         merged_rectangles.push_back(bound_rect[i]);
    //         total_area += bound_rect[i].area(); 
    // }

    // mean_area = (total_area/merged_rectangles.size());

    // Mat draw = Mat::ones(image.rows, image.cols, CV_32F) * 255;
    // Mat draw;
    // cvtColor(image, draw, COLOR_GRAY2BGR);

    // for (int i = 0; i < merged_rectangles.size(); i++){
    //     rectangle(draw, merged_rectangles[i].tl(), merged_rectangles[i].br(), Vec3b(0,0,255), 2, 8, 0);

        // if (merged_rectangles.size() < 20){
        //     if (merged_rectangles[i].area() >= best_area){
        //         best_area = merged_rectangles[i].area();

        //         min_x = merged_rectangles[i].tl().x;
        //         max_x = merged_rectangles[i].br().x;
        //         min_y = merged_rectangles[i].tl().y;
        //         max_y = merged_rectangles[i].br().y;
        //     }
        // } else if (
        //     merged_rectangles[i].width > merged_rectangles[i].height &&
        //     merged_rectangles[i].area() > mean_area*0.1 && 
        //     merged_rectangles[i].area() < mean_area*0.9
        // ){
        //     min_x = merged_rectangles[i].tl().x < min_x ? merged_rectangles[i].tl().x : min_x;
        //     min_y = merged_rectangles[i].tl().y < min_y ? merged_rectangles[i].tl().y : min_y;

        //     max_x = merged_rectangles[i].br().x > max_x ? merged_rectangles[i].br().x : max_x;
        //     max_y = merged_rectangles[i].br().y > max_y ? merged_rectangles[i].br().y : max_y;
        // }
    // }

    // Rect r = Rect(min_x, min_y, max_x-min_x, max_y-min_y);
    // rectangle(draw, r.tl(), r.br(), 0, 2, 8, 0);

    // output = draw; return;



	vector<vector<Point>> approx;
	approx.resize(contours.size());
	size_t i,j;

	for(i=0; i<contours.size(); i++){
		double peri = arcLength(contours[i], true);
		approxPolyDP(contours[i], approx[i], 0.02 * peri, true);
	}
	sort(approx.begin(), approx.end(), compareContourAreas);

    for(i = 0; i< approx.size(); i++){
		if(approx[i].size() == 4)
			break;
	}

    if(i < approx.size()){
        drawContours(output, approx, i, Scalar(0, 255, 0), 2);

		for(j=0; j<approx[i].size(); j++){
			approx[i][j] *= ratio;
		}
		// fourPointTransform(orig, output, approx[i]);
    }


}
