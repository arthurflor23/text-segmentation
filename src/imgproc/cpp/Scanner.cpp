#include "Scanner.hpp"

Scanner::Scanner() {
	this->cropped = false;
};

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

void resizeToHeight(Mat src, Mat &dst, int height){
	Size s = Size(src.cols * (height / double(src.rows)), height);
	resize(src, dst, s, INTER_AREA);
}

void Scanner::orderPoints(vector<Point> inpts, vector<Point> &ordered){
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

void Scanner::fourPointTransform(Mat src, Mat &dst, vector<Point> pts){
	vector<Point> orderedPts;
	orderPoints(pts, orderedPts);

	double wa = distance(orderedPts[2], orderedPts[3]);
	double wb = distance(orderedPts[1], orderedPts[0]);
	double mw = max(wa, wb);

	double ha = distance(orderedPts[1], orderedPts[2]);
	double hb = distance(orderedPts[0], orderedPts[3]);
	double mh = max(ha, hb);

	Point2f src_[] ={
        Point2f(orderedPts[0].x, orderedPts[0].y),
        Point2f(orderedPts[1].x, orderedPts[1].y),
        Point2f(orderedPts[2].x, orderedPts[2].y),
        Point2f(orderedPts[3].x, orderedPts[3].y),
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

void Scanner::processEdge(Mat input, Mat &output, int openKSize, int closeKSize, bool gaussianBlur){
    Mat image_pp, structuringElmt;
	cvtColor(input, image_pp, COLOR_BGR2GRAY);

	if (openKSize > 0){
		structuringElmt = getStructuringElement(MORPH_ELLIPSE, Size(openKSize,openKSize));
		morphologyEx(image_pp, image_pp, MORPH_OPEN, structuringElmt);
	}
	if (closeKSize > 0){
		structuringElmt = getStructuringElement(MORPH_ELLIPSE, Size(closeKSize,closeKSize));
		morphologyEx(image_pp, image_pp, MORPH_CLOSE, structuringElmt);
	}

	if (gaussianBlur){
		GaussianBlur(image_pp, image_pp, Size(7,7), 0);
	}
	Canny(image_pp, output, 50, 60, 3, true);
}

void Scanner::process(Mat image, Mat &output){
	Mat orig = image.clone();

	double ratio = image.rows / 500.0;
	resizeToHeight(image, image, 500);

	Mat edged, edgedCache;
	processEdge(image, edged, 11, 11, true);
	edgedCache = edged.clone();

	vector<vector<Point>> contours, shapes;
	vector<Vec4i> hierarchy;
	
	findContours(edged, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);
	edged = Mat::zeros(Size(image.cols, image.rows), CV_8UC1);

	vector<vector<Point>> hull(contours.size());
	int sum_area = 0, mean_area = 0;

	for(int i=0; i<contours.size(); i++){
		convexHull(Mat(contours[i]), hull[i], false);
		sum_area += contourArea(Mat(hull[i]));
	}
	mean_area = sum_area / hull.size();

	for(int i=0; i<hull.size(); i++){
		if(contourArea(Mat(hull[i])) >= mean_area){
			shapes.push_back(hull[i]);
		}
	}
	sort(shapes.begin(), shapes.end(), compareContourAreas);	

	for(int i=1; i<shapes.size(); i++){
		for(int j=0; j<shapes[i].size(); j++)
			shapes[0].push_back(shapes[i][j]);
	}

	convexHull(Mat(shapes[0]), hull[0], false);
	drawContours(edged, hull, 0, 255, 2);
	findContours(edged, contours, hierarchy, RETR_LIST, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> approx;
	approx.resize(contours.size());

	for(int i=0; i<contours.size(); i++){
		double peri = 0.01 * arcLength(contours[i], true);
		approxPolyDP(contours[i], approx[i], peri, true);
	}

	for(int i=0; i<approx.size(); i++){
		if(approx[i].size() == 4){
			for(int j=0; j<approx[i].size(); j++)
				approx[i][j] *= ratio;

			fourPointTransform(orig, output, approx[i]);
			this->cropped = true;
			return;
		}
	}

	processEdge(image, edgedCache, 101, 31, false);
    normalize(edgedCache, edgedCache, 0, 255, NORM_MINMAX, CV_32F);

	int minX = edgedCache.cols, minY = edgedCache.rows;
	int maxX = 0, maxY = 1;

	for (int i=0; i<edgedCache.rows; i++){
		for (int j=0; j<edgedCache.cols; j++){
			if (edgedCache.at<float>(i,j) > 0){
				minX = j < minX ? j : minX;
				minY = i < minY ? i : minY;

				maxX = j > maxX ? j : maxX;
				maxY = i > maxY ? i : maxY;
			}
		}
	}

	if (maxX > minX || maxY > minY){
		minX *= ratio; minY *= ratio;
		maxX *= ratio; maxY *= ratio;

		int width = maxX-minX;
		int height = maxY-minY;

		orig(Rect(minX, minY, width, height)).copyTo(output);
	} else {
		output = orig;
	}
}
