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

		foveatedHessianDetector(img1, Mat(), keypoints1, params);

		// computing descriptors
		SurfDescriptorExtractor extractor;
		Mat descriptors1;
		extractor.compute(img1, keypoints1, descriptors1);

		// drawing the results
		Mat outputImg;
		drawKeypoints(img1, keypoints1, outputImg, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	    imshow("keypoints", outputImg);
	    char key = waitKey(33);
		if(key == 'q') break;
		if(key == 'a') {
			params.foveaModel.growthfactor += 10;
		}
		if(key == 'd') {
			params.foveaModel.growthfactor = MAX(0, params.foveaModel.growthfactor-10);
		}
		if(key >= '1' && key <= '9') {
			if(key - '1' < params.foveaModel.beta.size())
				params.foveaModel.beta[key - '1'] = 1 - params.foveaModel.beta[key - '1'];
		}
	}

    return 0;
}
