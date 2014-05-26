
/*
   Copyright (C) 2014, Rafael Beserra <rafaelufrn@gmail.com>
   If you use this software for academic purposes, consider citing the related paper: Rafael Beserra Gomes, Bruno Motta de Carvalho, Luiz Marcos Garcia Gonçalves, Visual attention guided features selection with foveated images, Neurocomputing, Volume 120, 23 November 2013, Pages 34-44, ISSN 0925-2312, http://dx.doi.org/10.1016/j.neucom.2012.10.033.

   This file is part of foveatedFeatures software.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef FOVEATED_TRACKING
#define FOVEATED_TRACKING

#include "foveatedHessianDetector.h"
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include <iostream>

struct FoveatedTracking {

	Mat modelImg;
	vector<KeyPoint> modelKeypoints;
	vector<KeyPoint> imgKeypoints;
	vector<Point2f> obj_corners;
	vector<Point2f> scene_corners;
	vector<Point2f> modelPoints;
	vector<Point2f> imgPoints;
	Mat modelDescriptors;
	Mat imgDescriptors;
	String ymlFile;

	FoveatedHessianDetectorParams *params;
	int useFovea;

	//detection diagonals
	float diagonal1, diagonal2;
	float overallDistance;

	//object detected
	int detected;

	FoveatedTracking(Mat _modelImg, String _ymlFile) {
		useFovea = 1;
		_modelImg.copyTo(modelImg);
		ymlFile = _ymlFile;
		params = NULL;

		//model image feature extraction and descriptors
		SurfFeatureDetector detector(100);
		detector.detect(modelImg, modelKeypoints);
		SurfDescriptorExtractor extractor;
		extractor.compute(modelImg, modelKeypoints, modelDescriptors);

		//object corners points
		obj_corners.push_back(cvPoint(0, 0));
		obj_corners.push_back(cvPoint(modelImg.cols, 0));
		obj_corners.push_back(cvPoint(modelImg.cols, modelImg.rows));
		obj_corners.push_back(cvPoint(0, modelImg.rows));
	}

	int seemsToBeGood() {
		return fabs(diagonal1 - diagonal2) < 0.08*MAX(diagonal1, diagonal2) && overallDistance < 0.40;
	}

	void update(Mat frame, Mat &img_matches) {
		detected = 0;
		if(params == NULL) {
			params = new FoveatedHessianDetectorParams(frame.cols, frame.rows, ymlFile);
		}

		//Feature extraction and description
		if(useFovea)
			foveatedHessianDetector(frame, Mat(), imgKeypoints, *params);
		else {
			SurfFeatureDetector detector(300);
			detector.detect(frame, imgKeypoints);
		}
		SurfDescriptorExtractor extractor;
		extractor.compute(frame, imgKeypoints, imgDescriptors);

		//matching descriptors
		BFMatcher matcher(NORM_L2);
		vector<DMatch> matches;
		matcher.match(modelDescriptors, imgDescriptors, matches);
	
//		std::cout << matches.size() << " " << modelDescriptors.size() << " " << imgDescriptors.size() << std::endl;
		if(matches.size() > 5) {
			modelPoints.clear();
			imgPoints.clear();
			overallDistance = 0;
			for(int i = 0; i < matches.size(); i++) {
				DMatch m = matches[i];
				modelPoints.push_back(modelKeypoints[m.queryIdx].pt);
				imgPoints.push_back(imgKeypoints[m.trainIdx].pt);
				overallDistance += matches[i].distance;
			}
			overallDistance /= matches.size();
			if(modelPoints.size() > 5) {
				Mat H = findHomography(modelPoints, imgPoints, RANSAC, 4);
				perspectiveTransform(obj_corners, scene_corners, H);
				line(frame, scene_corners[0], scene_corners[1], Scalar(0, 255, 0), 4);
				line(frame, scene_corners[1], scene_corners[2], Scalar(0, 255, 0), 4);
				line(frame, scene_corners[2], scene_corners[3], Scalar(0, 255, 0), 4);
				line(frame, scene_corners[3], scene_corners[0], Scalar(0, 255, 0), 4);
				float dx = (scene_corners[2].x - scene_corners[0].x);
				float dy = (scene_corners[2].y - scene_corners[0].y);
				diagonal1 = sqrt(dx*dx + dy*dy);
				dx = (scene_corners[3].x - scene_corners[1].x);
				dy = (scene_corners[3].y - scene_corners[1].y);
				diagonal2 = sqrt(dx*dx + dy*dy);
				if(seemsToBeGood()) {
					detected = 1;
					float fx = (scene_corners[0].x + scene_corners[1].x + scene_corners[2].x + scene_corners[3].x)/4;
					float fy = (scene_corners[0].y + scene_corners[1].y + scene_corners[2].y + scene_corners[3].y)/4;
					params->foveaModel.setFovea(fx, fy);
				}
			}
		}
		if(!detected)
			useFovea = 0;
		else
			useFovea = 1;
		if(useFovea)
			drawFoveatedLevels(frame, *params);
		drawMatches(modelImg, modelKeypoints, frame, imgKeypoints, matches, img_matches, Scalar(0, 255, 255), Scalar(255, 255, 255), vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		//std::cout << overallDistance << std::endl;
	}

	~FoveatedTracking() {
		delete params;
	}

};


#endif

