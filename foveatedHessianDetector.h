

/*
   Copyright (C) 2014, Rafael Beserra <rafaelufrn@gmail.com>
   If you use this software for research purposes, consider citing the related paper: Rafael Beserra Gomes, Bruno Motta de Carvalho, Luiz Marcos Garcia Gonçalves, Visual attention guided features selection with foveated images, Neurocomputing, Volume 120, 23 November 2013, Pages 34-44, ISSN 0925-2312, http://dx.doi.org/10.1016/j.neucom.2012.10.033.

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
   
   This source code is based on SURF from Opencv-2.4.8. Below is the original copyright.
 */

/*
 * Copyright© 2008, Liu Liu All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 *  Redistributions of source code must retain the above
 *  copyright notice, this list of conditions and the following
 *  disclaimer.
 *  Redistributions in binary form must reproduce the above
 *  copyright notice, this list of conditions and the following
 *  disclaimer in the documentation and/or other materials
 *  provided with the distribution.
 *  The name of Contributor may not be used to endorse or
 *  promote products derived from this software without
 *  specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
 * TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 */

#ifndef FOVEATED_HESSIANDETECTOR
#define FOVEATED_HESSIANDETECTOR

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "linearFoveation.h"

using namespace cv;

static const int   SURF_ORI_SEARCH_INC = 5;
static const float SURF_ORI_SIGMA      = 2.5f;
static const float SURF_DESC_SIGMA     = 3.3f;

// Wavelet size at first layer of first octave.
static const int SURF_HAAR_SIZE0 = 9;

// Wavelet size increment between layers. This should be an even number,
// such that the wavelet sizes in an octave are either all even or all odd.
// This ensures that when looking for the neighbours of a sample, the layers
// above and below are aligned correctly.
static const int SURF_HAAR_SIZE_INC = 6;

//13/02/14 17:20:31 
//estrutura com os parâmetros para o detector hessiano foveado
struct FoveatedHessianDetectorParams {

	void init() {
		nOctaveLayers = 3;
		hessianThreshold = 100;
	}

	FoveatedHessianDetectorParams() {
		init();
	}

	FoveatedHessianDetectorParams(int imageWidth, int imageHeight, String ymlFile) {
		init();
		FileStorage fs(ymlFile, FileStorage::READ);
		foveaModel.wx = (int) fs["smallestLevelWidth"];
		foveaModel.wy = (int) fs["smallestLevelHeight"];
		
		fs["etavector"] >> foveaModel.eta;
		fs["bvector"] >> foveaModel.beta;
		fs["levelvector"] >> foveaModel.level;
		int numberOfLevels = (int) fs["numberOfLevels"];
		foveaModel.m = numberOfLevels - 1;
		fs["foveax"] >> foveaModel.fx;
		fs["foveay"] >> foveaModel.fy;
		fs["hessianThreshold"] >> hessianThreshold;
		fs.release();
		foveaModel.ux = imageWidth;
		foveaModel.uy = imageHeight;

		foveaModel.check();
		assert(hessianThreshold > 0);
		foveaModel.fixFovea();
	}

	int nOctaveLayers;
	float hessianThreshold;

	//foveation parameters
	LinearFoveation foveaModel;
};


struct SurfHF
{
	int p0, p1, p2, p3;
	float w;

	SurfHF(): p0(0), p1(0), p2(0), p3(0), w(0) {}
};

inline float calcHaarPattern( const int* origin, const SurfHF* f, int n )
{
	double d = 0;
	for( int k = 0; k < n; k++ )
		d += (origin[f[k].p0] + origin[f[k].p3] - origin[f[k].p1] - origin[f[k].p2])*f[k].w;
	return (float)d;
}

	static void
resizeHaarPattern( const int src[][5], SurfHF* dst, int n, int oldSize, int newSize, int widthStep )
{
	float ratio = (float)newSize/oldSize;
	for( int k = 0; k < n; k++ )
	{
		int dx1 = cvRound( ratio*src[k][0] );
		int dy1 = cvRound( ratio*src[k][1] );
		int dx2 = cvRound( ratio*src[k][2] );
		int dy2 = cvRound( ratio*src[k][3] );
		dst[k].p0 = dy1*widthStep + dx1;
		dst[k].p1 = dy2*widthStep + dx1;
		dst[k].p2 = dy1*widthStep + dx2;
		dst[k].p3 = dy2*widthStep + dx2;
		dst[k].w = src[k][4]/((float)(dx2-dx1)*(dy2-dy1));
	}
}

/*
 * Calculate the determinant and trace of the Hessian for a layer of the
 * scale-space pyramid
 */
static void calcLayerDetAndTrace( const Mat& sum, int size, int sampleStep,
		Mat& det, Mat& trace, FoveatedHessianDetectorParams params, int marginH, int foveaLevel )
{
	const int NX=3, NY=3, NXY=4;
	const int dx_s[NX][5] = { {0, 2, 3, 7, 1}, {3, 2, 6, 7, -2}, {6, 2, 9, 7, 1} };
	const int dy_s[NY][5] = { {2, 0, 7, 3, 1}, {2, 3, 7, 6, -2}, {2, 6, 7, 9, 1} };
	const int dxy_s[NXY][5] = { {1, 1, 4, 4, 1}, {5, 1, 8, 4, -1}, {1, 5, 4, 8, -1}, {5, 5, 8, 8, 1} };

	//foveated parameters
	int k = foveaLevel;

	int deltax = params.foveaModel.getDeltax(k);
	int deltay = params.foveaModel.getDeltay(k);
	int skx = params.foveaModel.getSizex(k);
	int sky = params.foveaModel.getSizey(k);

	//margin ref: centro da wavelet
	//margin_x ref: centro da wavelet
	int margin_x = MAX(marginH, deltax);
	int margin_y = MAX(marginH, deltay);

	//limit_x ref: centro da wavelet
	int limit_x = MIN(deltax + skx, params.foveaModel.ux - marginH);
	int limit_y = MIN(deltay + sky, params.foveaModel.uy - marginH);

	//sum_i ref: comeco da wavelet
	int sum_i, sum_j;
	sum_i = margin_y - size/2;

	//DEBUG
	/*
	std::cout << "Computando a imagem Hessiana" << std::endl;
	std::cout << "Margin H = " << marginH << std::endl;
	std::cout << "foveaLevel = " << foveaLevel << std::endl;
	std::cout << "fovea = " << fx << " " << fy << std::endl;
	std::cout << "delta = " << deltax << " " << deltay << std::endl;
	std::cout << "A wavelet vai de " << margin_x << " até " << limit_x << std::endl;
	std::cout << "Pulando de " << sampleStep << " em " << sampleStep << std::endl;
	*/

	SurfHF Dx[NX], Dy[NY], Dxy[NXY];

	if( size > sum.rows-1 || size > sum.cols-1 )
		return;

	resizeHaarPattern( dx_s , Dx , NX , 9, size, sum.cols );
	resizeHaarPattern( dy_s , Dy , NY , 9, size, sum.cols );
	resizeHaarPattern( dxy_s, Dxy, NXY, 9, size, sum.cols );

	for(int  i = 0; sum_i + size/2 <= limit_y; i++, sum_i += sampleStep ) {
		sum_j = margin_x - size/2;
		const int* sum_ptr = sum.ptr<int>(sum_i, sum_j);
		float* det_ptr = &det.at<float>(i, 0);
		float* trace_ptr = &trace.at<float>(i, 0);
		for(int j = 0; sum_j + size/2 <= limit_x; sum_j += sampleStep, j++ ) {
			float dx  = calcHaarPattern( sum_ptr, Dx , 3 );
			float dy  = calcHaarPattern( sum_ptr, Dy , 3 );
			float dxy = calcHaarPattern( sum_ptr, Dxy, 4 );
			sum_ptr += sampleStep;
			det_ptr[j] = dx*dy - 0.81f*dxy*dxy;
			trace_ptr[j] = dx + dy;
		}
	}
}


/*
 * Maxima location interpolation as described in "Invariant Features from
 * Interest Point Groups" by Matthew Brown and David Lowe. This is performed by
 * fitting a 3D quadratic to a set of neighbouring samples.
 *
 * The gradient vector and Hessian matrix at the initial keypoint location are
 * approximated using central differences. The linear system Ax = b is then
 * solved, where A is the Hessian, b is the negative gradient, and x is the
 * offset of the interpolated maxima coordinates from the initial estimate.
 * This is equivalent to an iteration of Netwon's optimisation algorithm.
 *
 * N9 contains the samples in the 3x3x3 neighbourhood of the maxima
 * dx is the sampling step in x
 * dy is the sampling step in y
 * ds is the sampling step in size
 * point contains the keypoint coordinates and scale to be modified
 *
 * Return value is 1 if interpolation was successful, 0 on failure.
 */
	static int
interpolateKeypoint( float N9[3][9], int dx, int dy, int ds, KeyPoint& kpt )
{
	Vec3f b(-(N9[1][5]-N9[1][3])/2,  // Negative 1st deriv with respect to x
			-(N9[1][7]-N9[1][1])/2,  // Negative 1st deriv with respect to y
			-(N9[2][4]-N9[0][4])/2); // Negative 1st deriv with respect to s

	Matx33f A(
			N9[1][3]-2*N9[1][4]+N9[1][5],            // 2nd deriv x, x
			(N9[1][8]-N9[1][6]-N9[1][2]+N9[1][0])/4, // 2nd deriv x, y
			(N9[2][5]-N9[2][3]-N9[0][5]+N9[0][3])/4, // 2nd deriv x, s
			(N9[1][8]-N9[1][6]-N9[1][2]+N9[1][0])/4, // 2nd deriv x, y
			N9[1][1]-2*N9[1][4]+N9[1][7],            // 2nd deriv y, y
			(N9[2][7]-N9[2][1]-N9[0][7]+N9[0][1])/4, // 2nd deriv y, s
			(N9[2][5]-N9[2][3]-N9[0][5]+N9[0][3])/4, // 2nd deriv x, s
			(N9[2][7]-N9[2][1]-N9[0][7]+N9[0][1])/4, // 2nd deriv y, s
			N9[0][4]-2*N9[1][4]+N9[2][4]);           // 2nd deriv s, s

	Vec3f x = A.solve(b, DECOMP_LU);

	bool ok = (x[0] != 0 || x[1] != 0 || x[2] != 0) &&
		std::abs(x[0]) <= 1 && std::abs(x[1]) <= 1 && std::abs(x[2]) <= 1;

	if( ok )
	{
		kpt.pt.x += x[0]*dx;
		kpt.pt.y += x[1]*dy;
		kpt.size = (float)cvRound( kpt.size + x[2]*ds );
	}
	return ok;
}

// Multi-threaded construction of the scale-space pyramid
struct SURFBuildInvoker : ParallelLoopBody
{
	SURFBuildInvoker( const Mat& _sum, const vector<int>& _sizes,
			const vector<int>& _sampleSteps,
			vector<Mat>& _dets, vector<Mat>& _traces, 
			FoveatedHessianDetectorParams _params,
			vector<int>& _margin, vector<int>& _foveaLevel)
	{
		sum = &_sum;
		sizes = &_sizes;
		sampleSteps = &_sampleSteps;
		dets = &_dets;
		traces = &_traces;
		params = _params;
		margin = &_margin;
		foveaLevel = &_foveaLevel;
	}

	void operator()(const Range& range) const
	{
		for( int i=range.start; i<range.end; i++ ) {
			if((*foveaLevel)[i] == -1) continue;
			calcLayerDetAndTrace( *sum, (*sizes)[i], (*sampleSteps)[i], (*dets)[i], (*traces)[i], params, (*margin)[i], (*foveaLevel)[i] );
		}
	}

	const Mat *sum;
	const vector<int> *sizes;
	const vector<int> *sampleSteps;
	const vector<int> *foveaLevel;
	const vector<int> *margin;
	vector<Mat>* dets;
	vector<Mat>* traces;
	FoveatedHessianDetectorParams params;
};

// Multi-threaded search of the scale-space pyramid for keypoints
struct SURFFindInvoker : ParallelLoopBody
{
	SURFFindInvoker( const Mat& _sum, const Mat& _mask_sum,
			const vector<Mat>& _dets, const vector<Mat>& _traces,
			const vector<int>& _sizes, const vector<int>& _sampleSteps,
			const vector<int>& _middleIndices, vector<KeyPoint>& _keypoints,
			int _nOctaveLayers, float _hessianThreshold,
			FoveatedHessianDetectorParams _params,
			vector<int>& _margin, vector<int>& _foveaLevel)
	{
		sum = &_sum;
		mask_sum = &_mask_sum;
		dets = &_dets;
		traces = &_traces;
		sizes = &_sizes;
		sampleSteps = &_sampleSteps;
		middleIndices = &_middleIndices;
		keypoints = &_keypoints;
		nOctaveLayers = _nOctaveLayers;
		hessianThreshold = _hessianThreshold;
		params = _params;
		margin = &_margin;
		foveaLevel = &_foveaLevel;
	}

	static void findMaximaInLayer( const Mat& sum, const Mat& mask_sum,
			const vector<Mat>& dets, const vector<Mat>& traces,
			const vector<int>& sizes, vector<KeyPoint>& keypoints,
			int octave, int layer, float hessianThreshold, int sampleStep,
			FoveatedHessianDetectorParams params, int marginH, int foveaLevel );

	void operator()(const Range& range) const
	{
		for( int i=range.start; i<range.end; i++ )
		{
			int layer = (*middleIndices)[i];
			int octave = params.foveaModel.eta[layer];
			if((*foveaLevel)[layer] == -1) continue;
			findMaximaInLayer( *sum, *mask_sum, *dets, *traces, *sizes,
					*keypoints, octave, layer, hessianThreshold,
					(*sampleSteps)[layer],
					params, (*margin)[layer], (*foveaLevel)[layer]);
		}
	}

	const Mat *sum;
	const Mat *mask_sum;
	const vector<Mat>* dets;
	const vector<Mat>* traces;
	const vector<int>* sizes;
	const vector<int>* sampleSteps;
	const vector<int>* middleIndices;
	vector<KeyPoint>* keypoints;
	int nOctaveLayers;
	float hessianThreshold;
	const vector<int> *foveaLevel;
	const vector<int> *margin;
	FoveatedHessianDetectorParams params;

	static Mutex findMaximaInLayer_m;
};

Mutex SURFFindInvoker::findMaximaInLayer_m;


/*
 * Find the maxima in the determinant of the Hessian in a layer of the
 * scale-space pyramid
 */
void SURFFindInvoker::findMaximaInLayer( const Mat& sum, const Mat& mask_sum,
		const vector<Mat>& dets, const vector<Mat>& traces,
		const vector<int>& sizes, vector<KeyPoint>& keypoints,
		int octave, int layer, float hessianThreshold, int sampleStep,
		FoveatedHessianDetectorParams params, int marginH, int foveaLevel )
{
	// Wavelet Data
	const int NM=1;
	const int dm[NM][5] = { {0, 0, 9, 9, 1} };
	SurfHF Dm;

	int size = sizes[layer];

	//foveated parameters
	int k = foveaLevel;

	int deltax = params.foveaModel.getDeltax(k);
	int deltay = params.foveaModel.getDeltay(k);
	int skx = params.foveaModel.getSizex(k);
	int sky = params.foveaModel.getSizey(k);

	//margin ref: centro da wavelet
	//margin_x ref: centro da wavelet
	int margin_x = MAX(marginH, deltax);
	int margin_y = MAX(marginH, deltay);

	//limit_x ref: centro da wavelet
	int limit_x = MIN(deltax + skx, params.foveaModel.ux - marginH);
	int limit_y = MIN(deltay + sky, params.foveaModel.uy - marginH);

	//sum_i ref: comeco da wavelet
	int sum_i, sum_j;
	sum_i = margin_y - size/2;

	//DEBUG
	/*
	std::cout << "Analisando o máximo da imagem Hessiana, na camada " << layer << std::endl;
	std::cout << "A wavelet vai de " << margin_x << " até " << limit_x << std::endl;
	std::cout << "Pulando de " << sampleStep << " em " << sampleStep << std::endl;
	*/

	if( !mask_sum.empty() )
		resizeHaarPattern( dm, &Dm, NM, 9, size, mask_sum.cols );

	int step = (int)(dets[layer].step/dets[layer].elemSize());

	for( int i = 0; sum_i + size/2 <= limit_y; i++, sum_i += sampleStep ) {
		sum_j = margin_x - size/2;
		const float* det_ptr = dets[layer].ptr<float>(i);
		const float* trace_ptr = traces[layer].ptr<float>(i);
		for(int j = 0; sum_j + size/2 <= limit_x; sum_j += sampleStep, j++ ) {
			float val0 = det_ptr[j];
			if(val0 > hessianThreshold) {
				/* The 3x3x3 neighbouring samples around the maxima.
				   The maxima is included at N9[1][4] */
				const float *det1 = &dets[layer-1].at<float>(i, j);
				const float *det2 = &dets[layer].at<float>(i, j);
				const float *det3 = &dets[layer+1].at<float>(i, j);
				float N9[3][9] = { { det1[-step-1], det1[-step], det1[-step+1],
					det1[-1]  , det1[0] , det1[1],
					det1[step-1] , det1[step] , det1[step+1]  },
					  { det2[-step-1], det2[-step], det2[-step+1],
						  det2[-1]  , det2[0] , det2[1],
						  det2[step-1] , det2[step] , det2[step+1]  },
					  { det3[-step-1], det3[-step], det3[-step+1],
						  det3[-1]  , det3[0] , det3[1],
						  det3[step-1] , det3[step] , det3[step+1]  } };

				/* Check the mask - why not just check the mask at the center of the wavelet? */
				if( !mask_sum.empty() )
				{
					const int* mask_ptr = &mask_sum.at<int>(sum_i, sum_j);
					float mval = calcHaarPattern( mask_ptr, &Dm, 1 );
					if( mval < 0.5 )
						continue;
				}

				/* Non-maxima suppression. val0 is at N9[1][4]*/
				if( val0 > N9[0][0] && val0 > N9[0][1] && val0 > N9[0][2] &&
						val0 > N9[0][3] && val0 > N9[0][4] && val0 > N9[0][5] &&
						val0 > N9[0][6] && val0 > N9[0][7] && val0 > N9[0][8] &&
						val0 > N9[1][0] && val0 > N9[1][1] && val0 > N9[1][2] &&
						val0 > N9[1][3]                    && val0 > N9[1][5] &&
						val0 > N9[1][6] && val0 > N9[1][7] && val0 > N9[1][8] &&
						val0 > N9[2][0] && val0 > N9[2][1] && val0 > N9[2][2] &&
						val0 > N9[2][3] && val0 > N9[2][4] && val0 > N9[2][5] &&
						val0 > N9[2][6] && val0 > N9[2][7] && val0 > N9[2][8] )
				{
					/* Calculate the wavelet center coordinates for the maxima */
					float center_i = sum_i + (size-1)*0.5f;
					float center_j = sum_j + (size-1)*0.5f;

					KeyPoint kpt( center_j, center_i, (float)sizes[layer],
							-1, val0, octave, CV_SIGN(trace_ptr[j]) );

					/* Interpolate maxima location within the 3x3x3 neighbourhood  */
					int ds = size - sizes[layer-1];
					int interp_ok = interpolateKeypoint( N9, sampleStep, sampleStep, ds, kpt );

					/* Sometimes the interpolation step gives a negative size etc. */
					if( interp_ok  )
					{
						/*printf( "KeyPoint %f %f %d\n", point.pt.x, point.pt.y, point.size );*/
						cv::AutoLock lock(findMaximaInLayer_m);
						keypoints.push_back(kpt);
					}
				}
			}
		}
	}
}

struct KeypointGreater
{
	inline bool operator()(const KeyPoint& kp1, const KeyPoint& kp2) const
	{
		if(kp1.response > kp2.response) return true;
		if(kp1.response < kp2.response) return false;
		if(kp1.size > kp2.size) return true;
		if(kp1.size < kp2.size) return false;
		if(kp1.octave > kp2.octave) return true;
		if(kp1.octave < kp2.octave) return false;
		if(kp1.pt.y < kp2.pt.y) return false;
		if(kp1.pt.y > kp2.pt.y) return true;
		return kp1.pt.x < kp2.pt.x;
	}
};


static void fastFoveatedHessianDetector( const Mat& sum, const Mat& mask_sum, vector<KeyPoint>& keypoints, FoveatedHessianDetectorParams params)
{
	/* Sampling step along image x and y axes at first octave. This is doubled
	   for each additional octave. WARNING: Increasing this improves speed,
	   however keypoint extraction becomes unreliable. */
	const int SAMPLE_STEP0 = 1;

	int nOctaveLayers = params.nOctaveLayers;
	float hessianThreshold = params.hessianThreshold;

	int nTotalLayers = (nOctaveLayers+2)*params.foveaModel.beta.size();
	int nMiddleLayers = nOctaveLayers*params.foveaModel.beta.size();

	vector<Mat> dets(nTotalLayers);
	vector<Mat> traces(nTotalLayers);
	vector<int> sizes(nTotalLayers);
	vector<int> sampleSteps(nTotalLayers);
	vector<int> middleIndices(nMiddleLayers);

	vector<int> foveaLevel(nTotalLayers);
	vector<int> margin(nTotalLayers);

	keypoints.clear();

	// Allocate space and calculate properties of each layer
	int index = 0, middleIndex = 0, step = SAMPLE_STEP0;

	for(unsigned int i = 0; i < params.foveaModel.beta.size(); i++) {
		for( int layer = 0; layer < nOctaveLayers+2; layer++ ) {
			/* The integral image sum is one pixel bigger than the source image*/
			margin[index] = ((SURF_HAAR_SIZE0+SURF_HAAR_SIZE_INC*(params.nOctaveLayers+1))<<(params.foveaModel.eta[i]-1))/2;
			if(params.foveaModel.beta[i] == 0)
				foveaLevel[index] = -1;
			else
				foveaLevel[index] = params.foveaModel.level[i];
			dets[index].create( params.foveaModel.uy, params.foveaModel.ux, CV_32F );
			traces[index].create( params.foveaModel.uy, params.foveaModel.ux, CV_32F );
			sizes[index] = (SURF_HAAR_SIZE0 + SURF_HAAR_SIZE_INC*layer) << (params.foveaModel.eta[i] - 1);
			sampleSteps[index] = 1 << (params.foveaModel.eta[i] - 1);

			if( 0 < layer && layer <= nOctaveLayers )
				middleIndices[middleIndex++] = index;
			//			std::cout << index << " " << layer << ", sampleStep = " << sampleSteps[index] << "\t";
			//			std::cout << "Size: " << sizes[index] << ", eta = " << params.eta[i] << std::endl;
			index++;
		}
	}

	params.foveaModel.check();
	// Calculate hessian determinant and trace samples in each layer
	parallel_for_( Range(0, nTotalLayers),
			SURFBuildInvoker(sum, sizes, sampleSteps, dets, traces, params, margin, foveaLevel) );

	// Find maxima in the determinant of the hessian
	parallel_for_( Range(0, nMiddleLayers),
			SURFFindInvoker(sum, mask_sum, dets, traces, sizes,
				sampleSteps, middleIndices, keypoints,
				nOctaveLayers, hessianThreshold, params, margin, foveaLevel) );

	std::sort(keypoints.begin(), keypoints.end(), KeypointGreater());
}


//13/02/14 17:18:19 
//função para aplicar detector hessiano foveado
static void foveatedHessianDetector(InputArray _img, InputArray _mask, vector<KeyPoint>& keypoints, FoveatedHessianDetectorParams params) {
	Mat sum, mask1, msum;
	Mat img = _img.getMat();
	Mat mask = _mask.getMat();

	params.foveaModel.check();
	integral(img, sum, CV_32S);
	if(!mask.empty()) {
		cv::min(mask, 1, mask1);
		integral(mask1, msum, CV_32S);
	}

	CV_Assert(!img.empty() && img.depth() == CV_8U);
	if( img.channels() > 1 )
		cvtColor(img, img, COLOR_BGR2GRAY);
	CV_Assert(params.hessianThreshold >= 0);
	CV_Assert(params.nOctaveLayers > 0);

	fastFoveatedHessianDetector(sum, msum, keypoints, params);
}

//função para desenhar
static void drawFoveatedLevels(InputArray _img, FoveatedHessianDetectorParams params) {
	params.foveaModel.check();
	Mat img = _img.getMat();
	for(int i = 0; i <= params.foveaModel.m; i++) {
		int dx = params.foveaModel.getDeltax(i);
		int dy = params.foveaModel.getDeltay(i);
		int sx = params.foveaModel.getSizex(i);
		int sy = params.foveaModel.getSizey(i);
		cv::rectangle(img, cv::Point(dx, dy), cv::Point(dx+sx, dy+sy), cv::Scalar(255, 255, 255));
	}
}


#endif

