#include "foveatedHessianDetector.h"
#include <stdio.h>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"

using namespace cv;

static void help()
{
    printf("\nThis program demonstrates using foveated features2d detector and descriptor extractor\n"
            "Using the SURF desriptor:\n"
            "\n"
            "Usage:\n extract <image1> <fovea yml file>\n");
}

static void on_mouse(int event, int x, int y, int flags, void *_param) {
	FoveatedHessianDetectorParams *params = (FoveatedHessianDetectorParams *) _param;
	params->foveaModel.setFovea(x, y);
	params->foveaModel.fixFovea();
}

int main(int argc, char** argv)
{
	int useFovea = 1;

    if(argc != 3)
    {
        help();
        return -1;
    }

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    if(img1.empty())
    {
        printf("Can't read one of the images\n");
        return -1;
    }

	FoveatedHessianDetectorParams params(img1.cols, img1.rows, argv[2]);
	namedWindow("keypoints", 1);
	cvSetMouseCallback("keypoints", &on_mouse, &params);

	while(true) {
		// detecting keypoints
		vector<KeyPoint> keypoints1;

		int64 t = cv::getTickCount();
		
		if(useFovea) {
			foveatedHessianDetector(img1, Mat(), keypoints1, params);
		} else {
			SurfFeatureDetector detector(400);
			detector.detect(img1, keypoints1);
		}
		t = cv::getTickCount() - t;
		std::cout << "Feature extraction = " << t*1000/cv::getTickFrequency() << std::endl;

		// computing descriptors
		SurfDescriptorExtractor extractor;
		Mat descriptors1;

		t = cv::getTickCount();
		extractor.compute(img1, keypoints1, descriptors1);
		t = cv::getTickCount() - t;
		std::cout << "Feature description = " << t*1000/cv::getTickFrequency() << std::endl;

		// drawing the results
		Mat outputImg;
		drawKeypoints(img1, keypoints1, outputImg, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		drawFoveatedLevels(outputImg, params);

	    imshow("keypoints", outputImg);

	    char key = waitKey(33);
		if(key == 'q') break;
		if(key == 'd') params.foveaModel.wx = MIN(params.foveaModel.ux-1, params.foveaModel.wx + 10);
		if(key == 'a') params.foveaModel.wx = MAX(1, params.foveaModel.wx-10);
		if(key == 'c') params.foveaModel.wy = MIN(params.foveaModel.uy-1, params.foveaModel.wy + 10);
		if(key == 'z') params.foveaModel.wy = MAX(1, params.foveaModel.wy-10);
		if(key >= '1' && key <= '9') {
			if(key - '1' < params.foveaModel.beta.size())
				params.foveaModel.beta[key - '1'] = 1 - params.foveaModel.beta[key - '1'];
		}
		if(key == 'f') useFovea = 1 - useFovea;
	}

    return 0;
}
