#include "../foveatedHessianDetector.h"
#include "../foveatedTracking.h"
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>

using namespace cv;

static void on_mouse(int event, int x, int y, int flags, void *_param) {
	FoveatedHessianDetectorParams *params = (FoveatedHessianDetectorParams *) _param;
	params->foveaModel.setFovea(x, y);
	params->foveaModel.fixFovea();
}

static void help()
{
    printf("\nThis program demonstrates using features2d detector, descriptor extractor and simple matcher\n"
            "Using the SURF desriptor:\n"
            "\n"
            "Usage:\n tracking <image1> <video device>\n");
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        help();
        return -1;
    }

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	VideoCapture cap;

    if(img1.empty()) {
        printf("Can't read one of the images\n");
        return -1;
    }

	cap.open(0);
    if( !cap.isOpened() ) {
		std::cout << "Could not initialize capturing...\n";
        return 0;
    }

	Mat image, img2;
	Mat img_matches;

	FoveatedTracking tracking(img1, "fovea1.yml");

	int firstFrame = 1;

    for(;;) {
        Mat frame;
        cap >> frame;
        if(frame.empty())
            break;

        frame.copyTo(image);
        cvtColor(image, img2, COLOR_BGR2GRAY);

		int64 t = cv::getTickCount();
		tracking.update(img2, img_matches);
		t = cv::getTickCount() - t;
		std::cout << t*1000/cv::getTickFrequency() << std::endl;

		if(firstFrame) {
			firstFrame = 0;
			namedWindow("matches", 1);
			cvSetMouseCallback("matches", &on_mouse, tracking.params);
		}

		imshow("matches", img_matches);

		FoveatedHessianDetectorParams *params = tracking.params;
        char key = (char)waitKey(10);
		if(key == 'q') break;
		if(key == 'd') params->foveaModel.wx = MIN(params->foveaModel.ux-1, params->foveaModel.wx + 10);
		if(key == 'a') params->foveaModel.wx = MAX(1, params->foveaModel.wx-10);
		if(key == 'c') params->foveaModel.wy = MIN(params->foveaModel.uy-1, params->foveaModel.wy + 10);
		if(key == 'z') params->foveaModel.wy = MAX(1, params->foveaModel.wy-10);
		if(key >= '1' && key <= '9') {
			if(key - '1' < params->foveaModel.beta.size())
				params->foveaModel.beta[key - '1'] = 1 - params->foveaModel.beta[key - '1'];
		}
		if(key == 'f') tracking.useFovea = 1 - tracking.useFovea;
	}


    return 0;
}
