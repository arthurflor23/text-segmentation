#include "Binarization.hpp"

#define uget(x,y)at<unsigned char>(y,x)
#define uset(x,y,v)at<unsigned char>(y,x)=v;
#define fget(x,y)at<float>(y,x)
#define fset(x,y,v)at<float>(y,x)=v;

Binarization::Binarization() {};

void Binarization::binarize(Mat image, Mat &output, int illumination, int option){
	cvtColor(image, this->grayscale, COLOR_BGR2GRAY);
	output = this->grayscale.clone();

    if (illumination) light_distribution();

    int winy = (int) (2.0 * this->grayscale.rows-1)/3;
    int winx = (int) this->grayscale.cols-1 < winy ? this->grayscale.cols-1 : winy;
    if (winx > 127) winx = winy = 127;

    if (option < 3){
        local_thresholding(this->grayscale, output, option, winx, winy, 0.1, 128);
    } else {
        otsu(this->grayscale, output);
    }
}

void Binarization::otsu(Mat grayscale, Mat &output){    
    Mat smoothed_img;
    blur(grayscale, smoothed_img, Size(3,3), Point(-1,-1));
    threshold(smoothed_img, output, 0.0, 255, THRESH_BINARY | THRESH_OTSU);
}

void Binarization::local_thresholding(Mat im, Mat &output, int option, int winx, int winy, double k, double dR){
	double m, s, max_s;
	double th = 0;
	double min_I, max_I;
	int wxh	= winx/2;
	int wyh	= winy/2;
	int x_firstth= wxh;
	int x_lastth = im.cols-wxh-1;
	int y_lastth = im.rows-wyh-1;
	int y_firstth = wyh;

	Mat map_m = Mat::zeros(im.rows, im.cols, CV_32F);
	Mat map_s = Mat::zeros(im.rows, im.cols, CV_32F);
	max_s = calc_local_stats(im, map_m, map_s, winx, winy);

	minMaxLoc(im, &min_I, &max_I);
	Mat thsurf(im.rows, im.cols, CV_32F);

	for	(int j=y_firstth; j<=y_lastth; j++){
		float *th_surf_data = thsurf.ptr<float>(j) + wxh;
		float *map_m_data = map_m.ptr<float>(j) + wxh;
		float *map_s_data = map_s.ptr<float>(j) + wxh;

		// NORMAL, NON-BORDER AREA IN THE MIDDLE OF THE WINDOW:
		for	(int i=0; i<=im.cols-winx; i++) {
			m = *map_m_data++;
			s = *map_s_data++;

    		switch (option) {
    			case 0: // NIBLACK
    				th = m + k*s;
    				break;

    			case 1: // SAUVOLA
	    			th = m * (1 + k*(s/dR-1));
	    			break;

    			case 2: // WOLF
    				th = m + k * (s/max_s-1) * (m-min_I);
    				break;
    		}

			*th_surf_data++ = th;

    		if (i==0){
        		// LEFT BORDER
				float *th_surf_ptr = thsurf.ptr<float>(j);
        		for (int i=0; i<=x_firstth; ++i)
					*th_surf_ptr++ = th;

        		// LEFT-UPPER CORNER
        		if (j==y_firstth){
        			for (int u=0; u<y_firstth; ++u){
						float *th_surf_ptr = thsurf.ptr<float>(u);
						for (int i=0; i<=x_firstth; ++i)
        					*th_surf_ptr++ = th;
					}
				}

        		// LEFT-LOWER CORNER
        		if (j == y_lastth){
        			for (int u=y_lastth+1; u<im.rows; ++u){
						float *th_surf_ptr = thsurf.ptr<float>(u);

        				for (int i=0; i<=x_firstth; ++i)
        					*th_surf_ptr++ = th;
					}
				}
    		}

			// UPPER BORDER
			if (j==y_firstth)
				for (int u=0; u<y_firstth; ++u)
					thsurf.fset(i+wxh,u,th);

			// LOWER BORDER
			if (j==y_lastth)
				for (int u=y_lastth+1; u<im.rows; ++u)
					thsurf.fset(i+wxh,u,th);
		}

		// RIGHT BORDER
		float *th_surf_ptr = thsurf.ptr<float>(j) + x_lastth;

		for (int i=x_lastth; i<im.cols; ++i)
			*th_surf_ptr++ = th;

  		// RIGHT-UPPER CORNER
		if (j==y_firstth){
			for (int u=0; u<y_firstth; ++u){
				float *th_surf_ptr = thsurf.ptr<float>(u) + x_lastth;

				for (int i=x_lastth; i<im.cols; ++i)
					*th_surf_ptr++ = th;
			}
		}

		// RIGHT-LOWER CORNER
		if (j==y_lastth){
			for (int u=y_lastth+1; u<im.rows; ++u){
				float *th_surf_ptr = thsurf.ptr<float>(u) + x_lastth;

				for (int i=x_lastth; i<im.cols; ++i)
					*th_surf_ptr++ = th;
			}
		}
	}

	for	(int y=0; y<im.rows; ++y){
		unsigned char *im_data = im.ptr<unsigned char>(y);
		float *th_surf_data = thsurf.ptr<float>(y);
		unsigned char *output_data = output.ptr<unsigned char>(y);

		for	(int x=0; x<im.cols; ++x){
			*output_data = *im_data >= *th_surf_data ? 255 : 0;
			im_data++;
			th_surf_data++;
			output_data++;
		}
	}
}

double Binarization::calc_local_stats(Mat &im, Mat &map_m, Mat &map_s, int winx, int winy){
    Mat im_sum, im_sum_sq;
    integral(im, im_sum, im_sum_sq, CV_64F);

	double m,s,max_s,sum,sum_sq;
	int wxh	= winx/2;
	int wyh	= winy/2;
	int x_firstth= wxh;
    int y_firstth= wyh;
	int y_lastth = im.rows-wyh-1;
	double winarea = winx*winy;

	max_s = 0;
	for	(int j = y_firstth ; j<=y_lastth; j++){
		sum = sum_sq = 0;

		double *sum_top_left = im_sum.ptr<double>(j - wyh);
		double *sum_top_right = sum_top_left + winx;
		double *sum_bottom_left = im_sum.ptr<double>(j - wyh + winy);
		double *sum_bottom_right = sum_bottom_left + winx;

		double *sum_eq_top_left = im_sum_sq.ptr<double>(j - wyh);
		double *sum_eq_top_right = sum_eq_top_left + winx;
		double *sum_eq_bottom_left = im_sum_sq.ptr<double>(j - wyh + winy);
		double *sum_eq_bottom_right = sum_eq_bottom_left + winx;

		sum = (*sum_bottom_right + *sum_top_left) - (*sum_top_right + *sum_bottom_left);
		sum_sq = (*sum_eq_bottom_right + *sum_eq_top_left) - (*sum_eq_top_right + *sum_eq_bottom_left);

		m  = sum / winarea;
		s  = sqrt ((sum_sq - m*sum)/winarea);
		if (s > max_s) max_s = s;

		float *map_m_data = map_m.ptr<float>(j) + x_firstth;
		float *map_s_data = map_s.ptr<float>(j) + x_firstth;
		*map_m_data++ = m;
		*map_s_data++ = s;

		for	(int i=1 ; i <= im.cols-winx; i++) {
			sum_top_left++, sum_top_right++, sum_bottom_left++, sum_bottom_right++;

			sum_eq_top_left++, sum_eq_top_right++, sum_eq_bottom_left++, sum_eq_bottom_right++;

			sum = (*sum_bottom_right + *sum_top_left) - (*sum_top_right + *sum_bottom_left);
			sum_sq = (*sum_eq_bottom_right + *sum_eq_top_left) - (*sum_eq_top_right + *sum_eq_bottom_left);

			m  = sum / winarea;
			s  = sqrt ((sum_sq - m*sum)/winarea);
			if (s > max_s) max_s = s;

			*map_m_data++ = m;
			*map_s_data++ = s;
		}
	}
	return max_s;
}

void Binarization::light_distribution(){
	get_histogram(this->grayscale);
	get_cei();
	get_edge();
	get_tli();

    Mat int_img = this->cei.clone();

    for (int y=0; y<int_img.cols; y++){
        for (int x=0; x<int_img.rows; x++){

            if (this->tli_erosion.at<float>(x,y) == 0){
                int head = x, end = x, n;

                while (end < this->tli_erosion.rows && this->tli_erosion.at<float>(end,y) == 0){
                    end++;
                }
                end--;
                n = end - head + 1;

                if (n <= 30){
                    vector<float> mpv_h, mpv_e;
                    double min_h, max_h, min_e, max_e;

                    for (int k=0; k<5; k++){
                        if ((head - k) >= 0)
                            mpv_h.push_back(this->cei.at<float>(head-k,y));
                        if ((end + k) < this->cei.rows)
                            mpv_e.push_back(this->cei.at<float>(end+k,y));
                    }

                    minMaxLoc(mpv_h, &min_h, &max_h);
                    minMaxLoc(mpv_e, &min_e, &max_e);

                    for (int m=0; m<n; m++)
                        int_img.at<float>(head+m,y) = max_h + (m+1) * ((max_e-max_h) / n);
                }
            }
        }
    }

    Mat kernel = Mat::ones(Size(11, 11), CV_32F) * 1/121;
    filter2D(scale(int_img), this->ldi, CV_32F, kernel);

    this->grayscale = (this->cei/this->ldi) * 260;

    for (int y=0; y<this->tli_erosion.rows; y++){
        for (int x=0; x<this->tli_erosion.cols; x++){
            if (this->tli_erosion.at<float>(y,x) != 0)
                this->grayscale.at<float>(y,x) *= 1.5;
        }
    }

    GaussianBlur(this->grayscale, this->grayscale, Size(3,3), 2);
    this->grayscale.convertTo(this->grayscale, CV_8U);
}

void Binarization::get_histogram(Mat image){
    vector<Mat> bgr_planes;
    split(image, bgr_planes);

    int histSize[] = {30};
    float bins[] = {0,300};
    const float *ranges[] = {bins};

    for (int i=0; i<bgr_planes.size(); i++){
        calcHist(&bgr_planes[i], 1, 0, Mat(), this->histogram, 1, histSize, ranges, true, true);
    }

    get_hr(sqrt(image.rows * image.cols));
}

void Binarization::get_hr(float sqrt_hw){
    this->hr = 0;
    for (int i=0; i<this->histogram.rows; i++){
        if (this->histogram.at<float>(i,0) > sqrt_hw){
            this->hr = (i * 10);
            break;
        }
    }
}

void Binarization::get_cei(){
    Mat cei = (this->grayscale - (this->hr + 50 * 0.4)) * 2;
    normalize(cei, this->cei, 0, 255, NORM_MINMAX, CV_32F);
    threshold(this->cei, this->cei_bin, 59, 255, THRESH_BINARY_INV);
}

void Binarization::get_edge(){
    float m1[] = {-1,0,1,-2,0,2,-1,0,1};
    float m2[] = {-2,-1,0,-1,0,1,0,1,2};
    float m3[] = {-1,-2,-1,0,0,0,1,2,1};
    float m4[] = {0,1,2,-1,0,1,-2,-1,0};

    Mat kernel1(3, 3, CV_32F, m1);
    Mat kernel2(3, 3, CV_32F, m2);
    Mat kernel3(3, 3, CV_32F, m3);
    Mat kernel4(3, 3, CV_32F, m4);

    Mat eg1, eg2, eg3, eg4;
    filter2D(this->grayscale, eg1, CV_32F, kernel1);
    eg1 = abs(eg1);

    filter2D(this->grayscale, eg2, CV_32F, kernel2);
    eg2 = abs(eg2);

    filter2D(this->grayscale, eg3, CV_32F, kernel3);
    eg3 = abs(eg3);

    filter2D(this->grayscale, eg4, CV_32F, kernel4);
    eg4 = abs(eg4);

    this->eg_avg = scale((eg1 + eg2 + eg3 + eg4)/4);
    threshold(this->eg_avg, this->eg_bin, 30, 255, THRESH_BINARY);
}

void Binarization::get_tli(){
    this->tli = Mat::ones(Size(this->grayscale.cols, this->grayscale.rows), CV_32F) * 255;
    this->tli -= this->eg_bin;
    this->tli -= this->cei_bin;
    threshold(this->tli, this->tli, 0, 255, THRESH_BINARY);

    Mat kernel = Mat::ones(Size(3, 3), CV_32F);
    erode(this->tli, this->tli_erosion, kernel);
    threshold(this->tli_erosion, this->tli_erosion, 0, 255, THRESH_BINARY);
}

Mat Binarization::scale(Mat image){
    double min, max;
    minMaxLoc(image, &min, &max);

    Mat res = image / (max - min);
    minMaxLoc(res, &min, &max);
    res -= min;
    res *= 255;

    return res;
}
