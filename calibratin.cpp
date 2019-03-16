#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <opencv2/ccalib/omnidir.hpp>
#include <opencv2/calib3d.hpp> 


using namespace cv;
using namespace std;

int main()
{
	vector< vector< Vec3f > > object_points;
	vector< vector< Vec2f > > image_points;
	vector< Vec2f > corners;
	int board_width = 7;
	int board_height = 5;
	int square_size = 29;
	cv::String path("E:/omnidir_image/*.jpg"); //select only jpg
	vector<cv::String> fn;
	vector<cv::Mat> data;
	cv::glob(path, fn, true); // recurse
	for (size_t k = 0; k < fn.size(); ++k)
	{
		Mat im = imread(fn[k], IMREAD_GRAYSCALE);
		if (im.empty()) continue; //only proceed if sucsessful
		// you probably want to do some preprocessing

		Size board_size = Size(board_width, board_height);
		int board_n = board_width * board_height;
		bool found = false;
		found = cv::findChessboardCorners(im, board_size, corners,
			CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_FAST_CHECK + CALIB_CB_NORMALIZE_IMAGE);
		if (found)
		{
			cornerSubPix(im, corners, cv::Size(7, 5), cv::Size(-1, -1),
				TermCriteria(TermCriteria::EPS | TermCriteria::MAX_ITER, 2000, 0.001));
			drawChessboardCorners(im, board_size, corners, found);
		}
		vector< Vec3f > obj;
		for (int i = 0; i < board_height; i++)
			for (int j = 0; j < board_width; j++)
				obj.push_back(Vec3f((float)j * square_size, (float)i * square_size, 0));

		if (found) {
			cout << k << ". Found corners!" << endl;
			image_points.push_back(corners);
			object_points.push_back(obj);
		}

		imshow("image", im);
		waitKey(200);
		data.push_back(im);

	}
	destroyAllWindows;

	cv::Mat K, xi, D, idx, k, d;

	int flags = cv::omnidir::CALIB_FIX_SKEW + cv::omnidir::CALIB_FIX_K1 + cv::omnidir::CALIB_FIX_K2;
	int flagf = cv::fisheye::CALIB_RECOMPUTE_EXTRINSIC | cv::fisheye::CALIB_FIX_SKEW;
	cv::TermCriteria critia(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 0.0001);
	//cv::TermCriteria critia2(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, DBL_EPSILON);

	std::vector<cv::Mat> rvecs, tvecs;

	//double rms = cv::fisheye::calibrate(object_points, image_points, cv::Size(1280, 960), k, d, rvecs, tvecs, flagf, critia2);
	double rms = cv::omnidir::calibrate(object_points, image_points, cv::Size(480, 480), K, xi, D, rvecs, tvecs, flags, critia, idx);


	print(K);
	print(D);
	print(xi);


	Mat dist = imread("E:/omnidir_image/1.jpg", IMREAD_COLOR);
	Mat undistorted, undistort;
	imshow("undistorted", dist);
	waitKey(1000);
	int flag1 = omnidir::RECTIFY_CYLINDRICAL;



	Matx33f Knew = Matx33f(370 / 3, 0, 220,
		0, 370 / 3, 240 / 7,
		0, 0, 1);

	Matx33f knew = Matx33f(1, 0, 0,
		0, 1, 0,
		0, 0, 1);

	omnidir::undistortImage(dist, undistorted, K, D, xi, flag1, Knew, cv::Size(780, 250));
	//fisheye::undistortImage(dist, undistort, k,d,k, cv::Size(1500, 800));

	//imshow("fish", undistort);

	imshow("undistorted", undistorted);
	waitKey(0);

	return 0;
}