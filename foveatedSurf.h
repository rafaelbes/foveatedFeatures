/* Original code has been submitted by Liu Liu. Here is the copyright.
----------------------------------------------------------------------------------
 * An OpenCV Implementation of SURF
 * Further Information Refer to "SURF: Speed-Up Robust Feature"
 * Author: Liu Liu
 * liuliu.1987+opencv@gmail.com
 *
 * There are still serveral lacks for this experimental implementation:
 * 1.The interpolation of sub-pixel mentioned in article was not implemented yet;
 * 2.A comparision with original libSurf.so shows that the hessian detector is not a 100% match to their implementation;
 * 3.Due to above reasons, I recommanded the original one for study and reuse;
 *
 * However, the speed of this implementation is something comparable to original one.
 *
 * Copyright© 2008, Liu Liu All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or
 * without modification, are permitted provided that the following
 * conditions are met:
 * 	Redistributions of source code must retain the above
 * 	copyright notice, this list of conditions and the following
 * 	disclaimer.
 * 	Redistributions in binary form must reproduce the above
 * 	copyright notice, this list of conditions and the following
 * 	disclaimer in the documentation and/or other materials
 * 	provided with the distribution.
 * 	The name of Contributor may not be used to endorse or
 * 	promote products derived from this software without
 * 	specific prior written permission.
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

/* 
   The following changes have been made, comparing to the original contribution:
   1. A lot of small optimizations, less memory allocations, got rid of global buffers
   2. Reversed order of cvGetQuadrangleSubPix and cvResize calls; probably less accurate, but much faster
   3. The descriptor computing part (which is most expensive) is threaded using OpenMP
   (subpixel-accurate keypoint localization and scale estimation are still TBD)
*/

/*
KeyPoint position and scale interpolation has been implemented as described in
the Brown and Lowe paper cited by the SURF paper.

The sampling step along the x and y axes of the image for the determinant of the
Hessian is now the same for each layer in an octave. While this increases the
computation time, it ensures that a true 3x3x3 neighbourhood exists, with
samples calculated at the same position in the layers above and below. This
results in improved maxima detection and non-maxima suppression, and I think it
is consistent with the description in the SURF paper.

The wavelet size sampling interval has also been made consistent. The wavelet
size at the first layer of the first octave is now 9 instead of 7. Along with
regular position sampling steps, this makes location and scale interpolation
easy. I think this is consistent with the SURF paper and original
implementation.

The scaling of the wavelet parameters has been fixed to ensure that the patterns
are symmetric around the centre. Previously the truncation caused by integer
division in the scaling ratio caused a bias towards the top left of the wavelet,
resulting in inconsistent keypoint positions.

The matrices for the determinant and trace of the Hessian are now reused in each
octave.

The extraction of the patch of pixels surrounding a keypoint used to build a
descriptor has been simplified.

KeyPoint descriptor normalisation has been changed from normalising each 4x4 
cell (resulting in a descriptor of magnitude 16) to normalising the entire 
descriptor to magnitude 1.

The default number of octaves has been increased from 3 to 4 to match the
original SURF binary default. The increase in computation time is minimal since
the higher octaves are sampled sparsely.

The default number of layers per octave has been reduced from 3 to 2, to prevent
redundant calculation of similar sizes in consecutive octaves.  This decreases 
computation time. The number of features extracted may be less, however the 
additional features were mostly redundant.

The radius of the circle of gradient samples used to assign an orientation has
been increased from 4 to 6 to match the description in the SURF paper. This is 
now defined by ORI_RADIUS, and could be made into a parameter.

The size of the sliding window used in orientation assignment has been reduced
from 120 to 60 degrees to match the description in the SURF paper. This is now
defined by ORI_WIN, and could be made into a parameter.

Other options like  HAAR_SIZE0, HAAR_SIZE_INC, SAMPLE_STEP0, ORI_SEARCH_INC, 
ORI_SIGMA and DESC_SIGMA have been separated from the code and documented. 
These could also be made into parameters.

Modifications by Ian Mahon

*/
#include "foveatedFeatures2d.hpp"
#include "foveatedSurfAux.h"
#include "precomp.hpp"


//const int foveafactor = 999999;
/* Wavelet size at first layer of first octave. */ 
const int HAAR_SIZE0 = 9;

/* Wavelet size increment between layers. This should be an even number,
 such that the wavelet sizes in an octave are either all even or all odd.
 This ensures that when looking for the neighbours of a sample, the layers
 above and below are aligned correctly. */
const int HAAR_SIZE_INC = 6;


void CvFoveatedSURFParams::print() {
	printf("numPatchElements = %d\n", numPatchElements);
	printf("elementSize = %d\n", elementSize);
	printf("descriptorBin = %d\n", descriptorBin);
	printf("patch size = %d\n", patch_size);
	printf("descriptor size = %d\n", descriptorSize);
	int index = 0;
	int layer;
	for(unsigned int i = 0; i < B.size(); i++) {
		printf("Nivel %d ----------------------- \n", i);
		printf("B[%d] = %d\n", i, B[i]);
		printf("eta[%d] = %d\n", i, eta[i]);
		for( layer=0; layer<nOctaveLayers+2; layer++ ) {
			printf("Camada %d: ", layer);
			printf("size = %d ", (HAAR_SIZE0+HAAR_SIZE_INC*layer)<<(eta[i]));
			printf("sample step = %d\n", 1<<eta[i]);
		}
		index++;
	}
}

void CvFoveatedSURFParams::addLevel(int _b, int _eta, int _level) {
	B.push_back(_b);
	eta.push_back(_eta);
	level.push_back(_level);
}

CvFoveatedSURFParams::CvFoveatedSURFParams(double _hessianThreshold, int _descriptorBin, int _numPatchElements, int _elementSize, int _wx, int _wy, int _m, int _foveafactor) {
    hessianThreshold = _hessianThreshold;
    upright = 0;
    nOctaves = 4;
    nOctaveLayers = 2;
	foveafactor = _foveafactor;

	wx = _wx;
	wy = _wy;
	m = _m;

	fx = fy = 0;

	//determinados por parametros
	numPatchElements = _numPatchElements;
	elementSize = _elementSize;
	descriptorBin = _descriptorBin; //8 or 4

	//calculados
	patch_size = elementSize*numPatchElements;
    descriptorSize = numPatchElements*numPatchElements*descriptorBin;
}

void CvFoveatedSURFParams::recalculate() {
	patch_size = elementSize*numPatchElements;
    descriptorSize = numPatchElements*numPatchElements*descriptorBin;
}

CvFoveatedSURFParams::CvFoveatedSURFParams() {
    hessianThreshold = 100;
    upright = 0;
    nOctaves = 4;
    nOctaveLayers = 2;
	foveafactor = 0;

	//determinados por parametros
	numPatchElements = 4;
	elementSize = 5;
	descriptorBin = 4; //8 or 4

	//calculados
	patch_size = elementSize*numPatchElements;
    descriptorSize = numPatchElements*numPatchElements*descriptorBin;
}

//revisto - aprimorar a questao dos indices
/*
 * Calculate the determinant and trace of the Hessian for a layer of the
 * scale-space pyramid
 */
CV_INLINE void
foveated_icvCalcLayerDetAndTrace( const CvMat* sum, int size, int sampleStep, CvMat *det, CvMat *trace, CvFoveatedSURFParams *params, int margin, int foveaLevel ) {
    const int NX=3, NY=3, NXY=4;
    const int dx_s[NX][5] = { {0, 2, 3, 7, 1}, {3, 2, 6, 7, -2}, {6, 2, 9, 7, 1} };
    const int dy_s[NY][5] = { {2, 0, 7, 3, 1}, {2, 3, 7, 6, -2}, {2, 6, 7, 9, 1} };
    const int dxy_s[NXY][5] = { {1, 1, 4, 4, 1}, {5, 1, 8, 4, -1}, {1, 5, 4, 8, -1}, {5, 5, 8, 8, 1} };

    foveatedCvSurfHF Dx[NX], Dy[NY], Dxy[NXY];
    double dx = 0, dy = 0, dxy = 0;
    int i, samples_i, samples_j ;
    int *sum_ptr;
    float *det_ptr, *trace_ptr;

    if( size>sum->rows-1 || size>sum->cols-1 )
       return;

	//usado para gravar as imagens dos pixels selecionados
	cv::Mat diagrama(sum->rows-1, sum->cols-1, CV_8UC1, cv::Scalar(255));

    foveated_icvResizeHaarPattern( dx_s , Dx , NX , 9, size, sum->cols );
    foveated_icvResizeHaarPattern( dy_s , Dy , NY , 9, size, sum->cols );
    foveated_icvResizeHaarPattern( dxy_s, Dxy, NXY, 9, size, sum->cols );

	int deltax, deltay;
	int limit_x, limit_y;
	int skx, sky;
	int k = foveaLevel;
	int m = params->m;
	int fx = params->fx;
	int fy = params->fy;
	int wx = params->wx;
	int wy = params->wy;
	int margin_x, margin_y;

	deltax = (k*(sum->cols-1 - wx + 2*fx))/(2*m);
	deltay = (k*(sum->rows-1 - wy + 2*fy))/(2*m);
	skx = (k*wx - k*(sum->cols-1) + m*(sum->cols-1))/m;
	sky = (k*wy - k*(sum->rows-1) + m*(sum->rows-1))/m;

	//margin ref: centro da wavelet
	//margin_x ref: centro da wavelet
	margin_x = MAX(margin, deltax - params->foveafactor);
	margin_y = MAX(margin, deltay - params->foveafactor);

	//limit_x ref: centro da wavelet
	limit_x = MIN(deltax + skx + params->foveafactor, sum->cols-1 - margin);
	limit_y = MIN(deltay + sky + params->foveafactor, sum->rows-1 - margin);
    /* The integral image 'sum' is one pixel bigger than the source image */
    samples_i = 1+(sum->rows-1-size)/sampleStep;
    samples_j = 1+(sum->cols-1-size)/sampleStep;

	//sum_i ref: comeco da wavelet
	int sum_i, sum_j;
	sum_i = margin_y - size/2;
	sum_j = margin_x - size/2;

//#define DIAGRAMAS 1
#ifdef DIAGRAMAS
	cv::rectangle(diagrama, cv::Point(sum_j, sum_i), cv::Point(sum_j + size, sum_i + size), cv::Scalar(150));
	cv::rectangle(diagrama, cv::Point(margin_x - margin, margin_y - margin), cv::Point(margin_x + margin, margin_y + margin), cv::Scalar(30), 2);
#endif

    for( i = 0; sum_i + size/2 <= limit_y; i++, sum_i += sampleStep ) {
		sum_j = margin_x - size/2;
        sum_ptr = sum->data.i + sum_i*sum->cols + sum_j;
        det_ptr = det->data.fl + i*det->cols;
        trace_ptr = trace->data.fl + i*trace->cols;
        for(; sum_j + size/2 <= limit_x; sum_j += sampleStep ) {
            dx  = foveated_icvCalcHaarPattern( sum_ptr, Dx , 3 );
            dy  = foveated_icvCalcHaarPattern( sum_ptr, Dy , 3 );
            dxy = foveated_icvCalcHaarPattern( sum_ptr, Dxy, 4 );
			diagrama.at<uchar>(sum_i + size/2, sum_j + size/2) = 0;
            sum_ptr += sampleStep;
            *det_ptr++ = (float)(dx*dy - 0.81*dxy*dxy);
            *trace_ptr++ = (float)(dx + dy);
        }
    }
#ifdef DIAGRAMAS
	printf("Saving diagrams\n");
	for(int p = 0; p <= m; p++) {
		deltax = (p*(sum->cols-1 - wx + 2*fx))/(2*m);
		deltay = (p*(sum->rows-1 - wy + 2*fy))/(2*m);
		skx = (p*wx - p*(sum->cols-1) + m*(sum->cols-1))/m;
		sky = (p*wy - p*(sum->rows-1) + m*(sum->rows-1))/m;
		cv::rectangle(diagrama, cv::Point(deltax, deltay), cv::Point(deltax + skx, deltay+ sky), cv::Scalar(150));
	}
	cv::line(diagrama, cv::Point(limit_x, 0), cv::Point(limit_x, sum->rows-1), cv::Scalar(30));
	cv::line(diagrama, cv::Point(0,limit_y), cv::Point(sum->cols-1, limit_y), cv::Scalar(30));
	char filename[30];
	sprintf(filename, "diagramas/c%04d.%04d.%04d.png", margin, size, foveaLevel);
	cv::imwrite(filename, diagrama);
#endif
}

/*
 * Find the maxima in the determinant of the Hessian in a layer of the 
 * scale-space pyramid
 */ 
CV_INLINE void foveated_icvFindMaximaInLayer( const CvMat *sum, const CvMat* mask_sum, const CvFoveatedSURFParams* params, CvMat **dets, CvMat **traces, const int *sizes, int layer, int sampleStep, CvSeq* points, int margin, int foveaLevel ) {
    /* Wavelet Data */
    const int NM=1;
    const int dm[NM][5] = { {0, 0, 9, 9, 1} };

    foveatedCvSurfHF Dm;
    int i, j, size, layer_rows, layer_cols;
    float *det_ptr, *trace_ptr;

    size = sizes[layer];

	int k = foveaLevel;
	int m = params->m;
	int fx = params->fx;
	int fy = params->fy;
	int wx = params->wx;
	int wy = params->wy;

	int deltax = (k*(sum->cols-1 - wx + 2*fx))/(2*m);
	int deltay = (k*(sum->rows-1 - wy + 2*fy))/(2*m);
	int skx = (k*wx - k*(sum->cols-1) + m*(sum->cols-1))/m;
	int sky = (k*wy - k*(sum->rows-1) + m*(sum->rows-1))/m;

	//margin ref: centro da wavelet
	//margin_x ref: centro da wavelet
	int margin_x = MAX(margin, deltax - params->foveafactor);
	int margin_y = MAX(margin, deltay - params->foveafactor);

	//limit_x ref: centro da wavelet
	int limit_x = MIN(deltax + skx + params->foveafactor, sum->cols-1 - margin);
	int limit_y = MIN(deltay + sky + params->foveafactor, sum->rows-1 - margin);

	//sum_i ref: comeco da wavelet
	int sum_i, sum_j;
	sum_i = margin_y - size/2;
	sum_j = margin_x - size/2;

    /* The integral image 'sum' is one pixel bigger than the source image */
    layer_rows = (sum->rows-1)/sampleStep;
    layer_cols = (sum->cols-1)/sampleStep;

    /* Ignore pixels without a 3x3x3 neighbourhood in the layer above */
//    margin = (sizes[layer+1]/2)/sampleStep+1; 

    if( mask_sum )
       foveated_icvResizeHaarPattern( dm, &Dm, NM, 9, size, mask_sum->cols );

    for( i = 0; sum_i + size/2 <= limit_y; i++, sum_i += sampleStep ) {
        //det_ptr = dets[layer]->data.fl + i*dets[layer]->cols;
        //trace_ptr = traces[layer]->data.fl + i*traces[layer]->cols;
		
		sum_j = margin_x - size/2;

        det_ptr = dets[layer]->data.fl + i*dets[layer]->cols;
        trace_ptr = traces[layer]->data.fl + i*traces[layer]->cols;
        for(j = 0; sum_j + size/2 <= limit_x; sum_j += sampleStep, j++ ) {
			if(i == 0 || j == 0 || sum_i + sampleStep > limit_y || sum_j + sampleStep > limit_x) continue;
            float val0 = det_ptr[j];
            if( val0 > params->hessianThreshold )
            {
                /* The 3x3x3 neighbouring samples around the maxima. 
                   The maxima is included at N9[1][4] */
                int c = dets[layer]->cols;
                const float *det1 = dets[layer-1]->data.fl + i*c + j;
                const float *det2 = dets[layer]->data.fl   + i*c + j;
                const float *det3 = dets[layer+1]->data.fl + i*c + j;
                float N9[3][9] = { { det1[-c-1], det1[-c], det1[-c+1],
                                     det1[-1]  , det1[0] , det1[1],
                                     det1[c-1] , det1[c] , det1[c+1]  },
                                   { det2[-c-1], det2[-c], det2[-c+1],
                                     det2[-1]  , det2[0] , det2[1],
                                     det2[c-1] , det2[c] , det2[c+1]  },
                                   { det3[-c-1], det3[-c], det3[-c+1],
                                     det3[-1]  , det3[0] , det3[1],
                                     det3[c-1] , det3[c] , det3[c+1]  } };

                /* Check the mask - why not just check the mask at the center of the wavelet? */
                if( mask_sum )
                {
                    const int* mask_ptr = mask_sum->data.i +  mask_sum->cols*sum_i + sum_j;
                    float mval = foveated_icvCalcHaarPattern( mask_ptr, &Dm, 1 );
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
                    double center_i = sum_i + (double)(size-1)/2;
                    double center_j = sum_j + (double)(size-1)/2;

                    CvSURFPoint point = cvSURFPoint( cvPoint2D32f(center_j,center_i),
                                                     CV_SIGN(trace_ptr[j]), sizes[layer], 0, val0 );

                    /* Interpolate maxima location within the 3x3x3 neighbourhood  */
                    int ds = size-sizes[layer-1];
                    int interp_ok = foveated_icvInterpolateKeypoint( N9, sampleStep, sampleStep, ds, &point );

                    /* Sometimes the interpolation step gives a negative size etc. */
                    if( interp_ok  )
                    {
                        /*printf( "KeyPoint %f %f %d\n", point.pt.x, point.pt.y, point.size );*/
                    #ifdef HAVE_TBB
                        static tbb::mutex m;
                        tbb::mutex::scoped_lock lock(m);
                    #endif
                        cvSeqPush( points, &point );
                    }
                }
            }
        }
    }
}


namespace cv
{
/* Multi-threaded construction of the scale-space pyramid */
struct SURFBuildInvoker {
    SURFBuildInvoker(const CvMat *_sum, const int *_sizes, const int *_sampleSteps,
                      CvMat** _dets, CvMat** _traces, CvFoveatedSURFParams *_params, const int *_margin, const int *_fovealevel) {
        sum = _sum;
        sizes = _sizes;
        sampleSteps = _sampleSteps;
        dets = _dets;
        traces = _traces;
		params = _params;
		foveaLevel = _fovealevel;
		margin = _margin;
    }

    void operator()(const BlockedRange& range) const
    {
        for( int i=range.begin(); i<range.end(); i++ ) {
            foveated_icvCalcLayerDetAndTrace(sum, sizes[i], sampleSteps[i], dets[i], traces[i], params, margin[i], foveaLevel[i]);
		}
    }

    const CvMat *sum;
    const int *sizes;
    const int *sampleSteps;
    const int *margin;
    const int *foveaLevel;
	CvFoveatedSURFParams *params;
    CvMat** dets;
    CvMat** traces;
};

/* Multi-threaded search of the scale-space pyramid for keypoints */
struct SURFFindInvoker
{
    SURFFindInvoker( const CvMat *_sum, const CvMat *_mask_sum, const CvFoveatedSURFParams* _params,
                     CvMat** _dets, CvMat** _traces,  const int *_sizes,
                     const int *_sampleSteps, const int *_middleIndices, CvSeq* _points, int *_margin, int *_foveaLevel )

    {
       sum = _sum;
       mask_sum = _mask_sum;
       params = _params;
       dets = _dets;
       traces = _traces;
       sizes = _sizes;
       sampleSteps = _sampleSteps;
       middleIndices = _middleIndices;
       points = _points;
	   foveaLevel = _foveaLevel;
	   margin = _margin;
    }

    void operator()(const BlockedRange& range) const
    {
        for( int i=range.begin(); i<range.end(); i++ )
        {
            int layer = middleIndices[i];
            foveated_icvFindMaximaInLayer( sum, mask_sum, params, dets, traces, sizes, layer, 
                                  sampleSteps[layer], points, margin[layer], foveaLevel[layer] );
        }
    }

    const CvMat *sum;
    const CvMat *mask_sum;
    const CvFoveatedSURFParams* params;
    const int *margin;
    const int *foveaLevel;
    CvMat** dets;
    CvMat** traces;
    const int *sizes;
    const int *sampleSteps;
    const int *middleIndices;
    CvSeq* points;
};

} // namespace cv


static CvSeq* icvFastHessianDetector( const CvMat* sum, const CvMat* mask_sum,
    CvMemStorage* storage, CvFoveatedSURFParams* params )
{
    CvSeq* points = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSURFPoint), storage );

    int nTotalLayers = (params->nOctaveLayers+2)*params->nOctaves;
    int nMiddleLayers = params->nOctaveLayers*params->nOctaves;
	nTotalLayers = (params->nOctaveLayers+2)*params->B.size();
	nMiddleLayers = (params->nOctaveLayers)*params->B.size();

    cv::AutoBuffer<CvMat*> dets(nTotalLayers);
    cv::AutoBuffer<CvMat*> traces(nTotalLayers);
    cv::AutoBuffer<int> sizes(nTotalLayers);
    cv::AutoBuffer<int> margin(nTotalLayers);
    cv::AutoBuffer<int> foveaLevel(nTotalLayers);
    cv::AutoBuffer<int> sampleSteps(nTotalLayers);
    cv::AutoBuffer<int> middleIndices(nMiddleLayers);
    int layer, step, index, middleIndex;

    /* Allocate space and calculate properties of each layer */
    index = 0;
    middleIndex = 0;
    step = 1;
	for(unsigned int i = 0; i < params->B.size(); i++) {
		for( layer=0; layer<params->nOctaveLayers+2; layer++ ) {
			margin[index] = ((HAAR_SIZE0+HAAR_SIZE_INC*(params->nOctaveLayers+1))<<(params->eta[i]))/2;
			foveaLevel[index] = params->level[i];
			dets[index]   = cvCreateMat( sum->rows-1, sum->cols-1, CV_32FC1 );
			traces[index] = cvCreateMat( sum->rows-1, sum->cols-1, CV_32FC1 );
			sizes[index] = (HAAR_SIZE0+HAAR_SIZE_INC*layer)<<(params->eta[i]);
			sampleSteps[index] = 1<<params->eta[i];
		//	printf("%d %d %d, margin = %d\n", (HAAR_SIZE0+HAAR_SIZE_INC*layer)<<(params->eta[i]), 1<<params->eta[i], index, margin[index]);
			if( layer!=0 && layer!=params->nOctaveLayers+1 )
				middleIndices[middleIndex++] = index;
			index++;
		}
	}
	//	printf("sampleStep[%d,%d] = %d\n", octave, layer, step);
	//	printf("sizes[%d,%d] = %d\n", octave, layer, sizes[index]);
	//	printf("\n");

#ifdef HAVE_TBB
    /* Calculate hessian determinant and trace samples in each layer*/
    cv::parallel_for( cv::BlockedRange(0, nTotalLayers),
                      cv::SURFBuildInvoker(sum,sizes,sampleSteps,dets,traces,params, margin, foveaLevel) );

    /* Find maxima in the determinant of the hessian */
    cv::parallel_for( cv::BlockedRange(0, nMiddleLayers),
                      cv::SURFFindInvoker(sum,mask_sum,params,dets,traces,sizes,
                                          sampleSteps,middleIndices,points, margin, foveaLevel) );
#else
    cv::SURFBuildInvoker(sum,sizes,sampleSteps,dets,traces,params,margin, foveaLevel)
	    (cv::BlockedRange(0, nTotalLayers));

    cv::SURFFindInvoker(sum,mask_sum,params,dets,traces,sizes, sampleSteps,middleIndices,points, margin, foveaLevel)
	    ( cv::BlockedRange(0, nMiddleLayers) );
#endif

    /* Clean-up */
    for( layer = 0; layer < nTotalLayers; layer++ )
    {
        cvReleaseMat( &dets[layer] );
        cvReleaseMat( &traces[layer] );
    }

    return points;
}


namespace cv
{

/* Methods to free data allocated in SURFInvoker constructor */
template<> inline void Ptr<float>::delete_obj()   { cvFree(&obj); }
template<> inline void Ptr<CvPoint>::delete_obj() { cvFree(&obj); }

struct SURFInvoker
{
    //enum { ORI_RADIUS = 6, ORI_WIN = 60, PATCH_SZ = 20 };
    enum { ORI_RADIUS = 6, ORI_WIN = 60};

    static const int   ORI_SEARCH_INC;
    static const float ORI_SIGMA;
    static const float DESC_SIGMA;

    SURFInvoker( const CvFoveatedSURFParams* _params,
                 CvSeq* _keypoints, CvSeq* _descriptors,
                 const CvMat* _img, const CvMat* _sum )
    {
        params = _params;
        keypoints = _keypoints;
        descriptors = _descriptors;
        img = _img;
        sum = _sum;

		int PATCH_SZ = _params->patch_size;

        /* Simple bound for number of grid points in circle of radius ORI_RADIUS */
        const int nOriSampleBound = (2*ORI_RADIUS+1)*(2*ORI_RADIUS+1);

        /* Allocate arrays */
        apt = (CvPoint*)cvAlloc(nOriSampleBound*sizeof(CvPoint));
        aptw = (float*)cvAlloc(nOriSampleBound*sizeof(float));
        DW = (float*)cvAlloc(PATCH_SZ*PATCH_SZ*sizeof(float));

        /* Coordinates and weights of samples used to calculate orientation */
        cv::Mat G_ori = cv::getGaussianKernel( 2*ORI_RADIUS+1, ORI_SIGMA, CV_32F );
        nOriSamples = 0;
        for( int i = -ORI_RADIUS; i <= ORI_RADIUS; i++ )
        {
            for( int j = -ORI_RADIUS; j <= ORI_RADIUS; j++ )
            {
                if( i*i + j*j <= ORI_RADIUS*ORI_RADIUS )
                {
                    apt[nOriSamples] = cvPoint(i,j);
                    aptw[nOriSamples++] = G_ori.at<float>(i+ORI_RADIUS,0) * G_ori.at<float>(j+ORI_RADIUS,0);
                }
            }
        }
        CV_Assert( nOriSamples <= nOriSampleBound );

        /* Gaussian used to weight descriptor samples */
        cv::Mat G_desc = cv::getGaussianKernel( PATCH_SZ, DESC_SIGMA, CV_32F );
        for( int i = 0; i < PATCH_SZ; i++ )
            for( int j = 0; j < PATCH_SZ; j++ )
                DW[i*PATCH_SZ+j] = G_desc.at<float>(i,0) * G_desc.at<float>(j,0);
    }

#define MAX_PATCH_SZ 50

    void operator()(const BlockedRange& range) const
    {
        /* X and Y gradient wavelet data */
        const int NX=2, NY=2;
        const int dx_s[NX][5] = {{0, 0, 2, 4, -1}, {2, 0, 4, 4, 1}};
        const int dy_s[NY][5] = {{0, 0, 4, 2, 1}, {0, 2, 4, 4, -1}};

        const int descriptor_size = params->descriptorSize;
		
        /* Optimisation is better using nOriSampleBound than nOriSamples for 
         array lengths.  Maybe because it is a constant known at compile time */
        const int nOriSampleBound =(2*ORI_RADIUS+1)*(2*ORI_RADIUS+1);

		int PATCH_SZ = params->patch_size;

        float X[nOriSampleBound], Y[nOriSampleBound], angle[nOriSampleBound];
//        uchar PATCH[PATCH_SZ+1][PATCH_SZ+1];

        float DX[MAX_PATCH_SZ][MAX_PATCH_SZ], DY[MAX_PATCH_SZ][MAX_PATCH_SZ];
        CvMat matX = cvMat(1, nOriSampleBound, CV_32F, X);
        CvMat matY = cvMat(1, nOriSampleBound, CV_32F, Y);
        CvMat _angle = cvMat(1, nOriSampleBound, CV_32F, angle);
        Mat _patch(PATCH_SZ+1, PATCH_SZ+1, CV_8U);

        int k, k1 = range.begin(), k2 = range.end();
        int maxSize = 0;
        for( k = k1; k < k2; k++ )
        {
            maxSize = std::max(maxSize, ((CvSURFPoint*)cvGetSeqElem( keypoints, k ))->size);
        }
        maxSize = cvCeil((PATCH_SZ+1)*maxSize*1.2f/9.0f);
        Ptr<CvMat> winbuf = cvCreateMat( 1, maxSize > 0 ? maxSize*maxSize : 1, CV_8U );
        for( k = k1; k < k2; k++ )
        {
            const int* sum_ptr = sum->data.i;
            int sum_cols = sum->cols;
            int i, j, kk, x, y, nangle;
            float* vec;
            foveatedCvSurfHF dx_t[NX], dy_t[NY];
            CvSURFPoint* kp = (CvSURFPoint*)cvGetSeqElem( keypoints, k );
            int size = kp->size;
            CvPoint2D32f center = kp->pt;
            /* The sampling intervals and wavelet sized for selecting an orientation
             and building the keypoint descriptor are defined relative to 's' */
            float s = (float)size*1.2f/9.0f;
            /* To find the dominant orientation, the gradients in x and y are
             sampled in a circle of radius 6s using wavelets of size 4s.
             We ensure the gradient wavelet size is even to ensure the 
             wavelet pattern is balanced and symmetric around its center */
            int grad_wav_size = 2*cvRound( 2*s );
            if ( sum->rows < grad_wav_size || sum->cols < grad_wav_size )
            {
                /* when grad_wav_size is too big,
                 * the sampling of gradient will be meaningless
                 * mark keypoint for deletion. */
                kp->size = -1;
                continue;
            }

            float descriptor_dir = 90.f;
            if (params->upright == 0)
            {
                foveated_icvResizeHaarPattern( dx_s, dx_t, NX, 4, grad_wav_size, sum->cols );
                foveated_icvResizeHaarPattern( dy_s, dy_t, NY, 4, grad_wav_size, sum->cols );
                for( kk = 0, nangle = 0; kk < nOriSamples; kk++ )
                {
                    const int* ptr;
                    float vx, vy;
                    x = cvRound( center.x + apt[kk].x*s - (float)(grad_wav_size-1)/2 );
                    y = cvRound( center.y + apt[kk].y*s - (float)(grad_wav_size-1)/2 );
                    if( (unsigned)y >= (unsigned)(sum->rows - grad_wav_size) ||
                        (unsigned)x >= (unsigned)(sum->cols - grad_wav_size) )
                        continue;
                    ptr = sum_ptr + x + y*sum_cols;
                    vx = foveated_icvCalcHaarPattern( ptr, dx_t, 2 );
                    vy = foveated_icvCalcHaarPattern( ptr, dy_t, 2 );
                    X[nangle] = vx*aptw[kk]; Y[nangle] = vy*aptw[kk];
                    nangle++;
                }
                if ( nangle == 0 )
                {
                    /* No gradient could be sampled because the keypoint is too
                    * near too one or more of the sides of the image. As we
                    * therefore cannot find a dominant direction, we skip this
                    * keypoint and mark it for later deletion from the sequence. */
                    kp->size = -1;
                    continue;
                }
                matX.cols = matY.cols = _angle.cols = nangle;
                cvCartToPolar( &matX, &matY, 0, &_angle, 1 );

                float bestx = 0, besty = 0, descriptor_mod = 0;
                for( i = 0; i < 360; i += ORI_SEARCH_INC )
                {
                    float sumx = 0, sumy = 0, temp_mod;
                    for( j = 0; j < nangle; j++ )
                    {
                        int d = std::abs(cvRound(angle[j]) - i);
                        if( d < ORI_WIN/2 || d > 360-ORI_WIN/2 )
                        {
                            sumx += X[j];
                            sumy += Y[j];
                        }
                    }
                    temp_mod = sumx*sumx + sumy*sumy;
                    if( temp_mod > descriptor_mod )
                    {
                        descriptor_mod = temp_mod;
                        bestx = sumx;
                        besty = sumy;
                    }
                }
                descriptor_dir = cvFastArctan( besty, bestx );
            }
            kp->dir = descriptor_dir;
            if( !descriptors )
                continue;
            
            /* Extract a window of pixels around the keypoint of size 20s */
            int win_size = (int)((PATCH_SZ+1)*s);
            CV_Assert( winbuf->cols >= win_size*win_size );
            //CvMat win = cvMat(win_size, win_size, CV_8U, winbuf->data.ptr);
            Mat win(win_size, win_size, CV_8U, winbuf->data.ptr);

            if (params->upright == 0)
            {
            	descriptor_dir *= (float)(CV_PI/180);
                float sin_dir = sin(descriptor_dir);
                float cos_dir = cos(descriptor_dir);

                /* Subpixel interpolation version (slower). Subpixel not required since
                the pixels will all get averaged when we scale down to 20 pixels */
                /*  
                float w[] = { cos_dir, sin_dir, center.x,
                -sin_dir, cos_dir , center.y };
                CvMat W = cvMat(2, 3, CV_32F, w);
                cvGetQuadrangleSubPix( img, &win, &W );
                */

                /* Nearest neighbour version (faster) */
                float win_offset = -(float)(win_size-1)/2;
                float start_x = center.x + win_offset*cos_dir + win_offset*sin_dir;
                float start_y = center.y - win_offset*sin_dir + win_offset*cos_dir;
                //uchar* WIN = win.data.ptr;
                for( i = 0; i < win_size; i++, start_x += sin_dir, start_y += cos_dir ) {
                    float pixel_x = start_x;
                    float pixel_y = start_y;
                    for( j = 0; j < win_size; j++, pixel_x += cos_dir, pixel_y -= sin_dir ) {
                        int x = std::min(std::max(cvRound(pixel_x), 0), img->cols-1);
                        int y = std::min(std::max(cvRound(pixel_y), 0), img->rows-1);
                        //WIN[i*win_size + j] = img->data.ptr[y*img->step + x];
                        win.at<uchar>(i, j) = img->data.ptr[y*img->step + x];
                    }
                }
            }
            else
            {
                /* extract rect - slightly optimized version of the code above
                   TODO: find faster code, as this is simply an extract rect operation, 
                         e.g. by using cvGetSubRect, problem is the border processing */
                // descriptor_dir == 90 grad
                // sin_dir == 1
                // cos_dir == 0
                
                float win_offset = -(float)(win_size-1)/2;
                int start_x = cvRound(center.x + win_offset);
                int start_y = cvRound(center.y - win_offset);
                //uchar* WIN = win.data.ptr;
                for( i = 0; i < win_size; i++, start_x++ )
                {
                    int pixel_x = start_x;
                    int pixel_y = start_y;
                    for( j=0; j<win_size; j++, pixel_y-- )
                    {
                        x = MAX( pixel_x, 0 );
                        y = MAX( pixel_y, 0 );
                        x = MIN( x, img->cols-1 );
                        y = MIN( y, img->rows-1 );
                        //WIN[i*win_size + j] = img->data.ptr[y*img->step+x];
                        win.at<uchar>(i, j) = img->data.ptr[y*img->step + x];
                    }
                }               
            }
            /* Scale the window to size PATCH_SZ so each pixel's size is s. This
             makes calculating the gradients with wavelets of size 2s easy */
            //cvResize( &win, &_patch, CV_INTER_AREA );
            resize( win, _patch, _patch.size(), 0, 0, CV_INTER_AREA );

            /* Calculate gradients in x and y with wavelets of size 2s */
            for( i = 0; i < PATCH_SZ; i++ )
                for( j = 0; j < PATCH_SZ; j++ )
                {
                    float dw = DW[i*PATCH_SZ + j];
                    //float vx = (PATCH[i][j+1] - PATCH[i][j] + PATCH[i+1][j+1] - PATCH[i+1][j])*dw; //sub
                    //float vy = (PATCH[i+1][j] - PATCH[i][j] + PATCH[i+1][j+1] - PATCH[i][j+1])*dw; //sub
                    float vx = (_patch.at<uchar>(i, j+1) - _patch.at<uchar>(i,j) + _patch.at<uchar>(i+1,j+1) - _patch.at<uchar>(i+1,j))*dw;
                    float vy = (_patch.at<uchar>(i+1, j) - _patch.at<uchar>(i,j) + _patch.at<uchar>(i+1,j+1) - _patch.at<uchar>(i,j+1))*dw;
                    DX[i][j] = vx;
                    DY[i][j] = vy;
                }

            /* Construct the descriptor */
            vec = (float*)cvGetSeqElem( descriptors, k );
//			printf("Descriptor limit = %d (%d/%d)\n", (int)(descriptors->elem_size/sizeof(vec[0])), descriptors->elem_size, sizeof(vec[0]));
            for( kk = 0; kk < (int)(descriptors->elem_size/sizeof(vec[0])); kk++ )
                vec[kk] = 0;
            double square_mag = 0;
			int numPatchElements = params->numPatchElements;
			int elementSize = params->elementSize;
			int descriptorBin = params->descriptorBin;

			//descriptor k
//#define SAVEPTCH 1
#ifdef SAVEPTCH
			char filename[30];
			sprintf(filename, "patches/p%03d-%04d.png",size, k);
			imwrite(filename, _patch);
#endif

            if(params->descriptorBin == 8) {
                for( i = 0; i < numPatchElements; i++ )
                    for( j = 0; j < numPatchElements; j++ ) {
                        for( y = i*elementSize; y < i*elementSize+elementSize; y++ )
                            for( x = j*elementSize; x < j*elementSize+elementSize; x++ ) {
                                float tx = DX[y][x], ty = DY[y][x];
                                if( ty >= 0 ) {
                                    vec[0] += tx;
                                    vec[1] += (float)fabs(tx);
                                } else {
                                    vec[2] += tx;
                                    vec[3] += (float)fabs(tx);
                                }
                                if ( tx >= 0 ) {
                                    vec[4] += ty;
                                    vec[5] += (float)fabs(ty);
                                } else {
                                    vec[6] += ty;
                                    vec[7] += (float)fabs(ty);
                                }
                            }
                        for( kk = 0; kk < 8; kk++ )
                            square_mag += vec[kk]*vec[kk];
                        vec += 8;
                    }
            } else if(params->descriptorBin == 4) {
                for( i = 0; i < numPatchElements; i++ )
                    for( j = 0; j < numPatchElements; j++ ) {
                        for( y = i*elementSize; y < i*elementSize+elementSize; y++ )
                            for( x = j*elementSize; x < j*elementSize+elementSize; x++ ) {
                                float tx = DX[y][x], ty = DY[y][x];
                                vec[0] += tx; vec[1] += ty;
                                vec[2] += (float)fabs(tx); vec[3] += (float)fabs(ty);
                            }
                        for( kk = 0; kk < descriptorBin; kk++ )
                            square_mag += vec[kk]*vec[kk];
                        vec+=4;
                    }
            } else if(params->descriptorBin == 2) {
                for( i = 0; i < numPatchElements; i++ )
                    for( j = 0; j < numPatchElements; j++ ) {
                        for( y = i*elementSize; y < i*elementSize+elementSize; y++ )
                            for( x = j*elementSize; x < j*elementSize+elementSize; x++ ) {
                                float tx = DX[y][x], ty = DY[y][x];
                                vec[0] += tx; vec[1] += ty;
                            }
                        for( kk = 0; kk < descriptorBin; kk++ )
                            square_mag += vec[kk]*vec[kk];
                        vec+=2;
                    }
            } else {
				printf("No descriptor bin\n");
			}

            /* unit vector is essential for contrast invariance */
            vec = (float*)cvGetSeqElem( descriptors, k );
            double scale = 1./(sqrt(square_mag) + DBL_EPSILON);
            for( kk = 0; kk < descriptor_size; kk++ )
                vec[kk] = (float)(vec[kk]*scale);
        }
    }

    /* Parameters */
    const CvFoveatedSURFParams* params;
    const CvMat* img;
    const CvMat* sum;
    CvSeq* keypoints;
    CvSeq* descriptors;

    /* Pre-calculated values */
    int nOriSamples;
    cv::Ptr<CvPoint> apt;
    cv::Ptr<float> aptw;
    cv::Ptr<float> DW;
};

const int   SURFInvoker::ORI_SEARCH_INC = 5;
const float SURFInvoker::ORI_SIGMA      = 2.5f;
const float SURFInvoker::DESC_SIGMA     = 3.3f;
}


CV_IMPL void
cvExtractFoveatedSURF( const CvArr* _img, const CvArr* _mask,
               CvSeq** _keypoints, CvSeq** _descriptors,
               CvMemStorage* storage, CvFoveatedSURFParams params,
               int useProvidedKeyPts)
{
    CvMat *sum = 0, *mask1 = 0, *mask_sum = 0;

    if( _keypoints && !useProvidedKeyPts ) // If useProvidedKeyPts!=0 we'll use current contents of "*_keypoints"
        *_keypoints = 0;
    if( _descriptors )
        *_descriptors = 0;

    CvSeq *keypoints, *descriptors = 0;
    CvMat imghdr, *img = cvGetMat(_img, &imghdr);
    CvMat maskhdr, *mask = _mask ? cvGetMat(_mask, &maskhdr) : 0;

    int descriptor_size = params.descriptorSize;
    const int descriptor_data_type = CV_32F;
    int i, N;

    CV_Assert(img != 0);
    CV_Assert(CV_MAT_TYPE(img->type) == CV_8UC1);
    CV_Assert(mask == 0 || (CV_ARE_SIZES_EQ(img,mask) && CV_MAT_TYPE(mask->type) == CV_8UC1));
    CV_Assert(storage != 0);
    //CV_Assert(params.hessianThreshold >= 0);
    CV_Assert(params.nOctaves > 0);
    CV_Assert(params.nOctaveLayers > 0);

    sum = cvCreateMat( img->rows+1, img->cols+1, CV_32SC1 );
    cvIntegral( img, sum );

    // Compute keypoints only if we are not asked for evaluating the descriptors are some given locations:
    if (!useProvidedKeyPts)
    {
        if( mask )
        {
            mask1 = cvCreateMat( img->height, img->width, CV_8UC1 );
            mask_sum = cvCreateMat( img->height+1, img->width+1, CV_32SC1 );
            cvMinS( mask, 1, mask1 );
            cvIntegral( mask1, mask_sum );
        }
        keypoints = icvFastHessianDetector( sum, mask_sum, storage, &params );
    }
    else
    {
        CV_Assert(useProvidedKeyPts && (_keypoints != 0) && (*_keypoints != 0));
        keypoints = *_keypoints;
    }

    N = keypoints->total;
    if( _descriptors )
    {
        descriptors = cvCreateSeq( 0, sizeof(CvSeq),
            descriptor_size*CV_ELEM_SIZE(descriptor_data_type), storage );
        cvSeqPushMulti( descriptors, 0, N );
    }


    if ( N > 0 )
    {
#ifdef HAVE_TBB
        cv::parallel_for(cv::BlockedRange(0, N),
                     cv::SURFInvoker(&params, keypoints, descriptors, img, sum) );
#else
	    cv::SURFInvoker invoker(&params, keypoints, descriptors, img, sum);
	    invoker(cv::BlockedRange(0, N));
#endif
    }


    /* remove keypoints that were marked for deletion */
    for ( i = 0; i < N; i++ )
    {
        CvSURFPoint* kp = (CvSURFPoint*)cvGetSeqElem( keypoints, i );
        if ( kp->size == -1 )
        {
            cvSeqRemove( keypoints, i );
            if ( _descriptors )
                cvSeqRemove( descriptors, i );
            i--;
            N--;
        }
    }

    if( _keypoints && !useProvidedKeyPts )
        *_keypoints = keypoints;
    if( _descriptors )
        *_descriptors = descriptors;

    cvReleaseMat( &sum );
    if (mask1) cvReleaseMat( &mask1 );
    if (mask_sum) cvReleaseMat( &mask_sum );
}


namespace cv
{

FoveatedSURF::FoveatedSURF()
{
    hessianThreshold = 100;
    upright = 0;
    nOctaves = 4;
    nOctaveLayers = 2;

	numPatchElements = 4;
	elementSize = 5;
	descriptorBin = 4; //8 or 4

	//calculados
	patch_size = elementSize*numPatchElements;
    descriptorSize = numPatchElements*numPatchElements*descriptorBin;
		printf("descriptor size = %d\n", descriptorSize);
}

FoveatedSURF::FoveatedSURF(double _threshold, int _nOctaves, int _nOctaveLayers, bool _upright)
{
    hessianThreshold = _threshold;
    upright = _upright;
    nOctaves = _nOctaves;
    nOctaveLayers = _nOctaveLayers;
	
	numPatchElements = 4;
	elementSize = 5;
	descriptorBin = 4; //8 or 4

	//calculados
	patch_size = elementSize*numPatchElements;
    descriptorSize = numPatchElements*numPatchElements*descriptorBin;
		//printf("descriptor size = %d\n", descriptorSize);
}

int FoveatedSURF::getDescriptorSize() const { return descriptorSize; }

static int getPointOctave(const CvSURFPoint& kpt, const CvFoveatedSURFParams& params)
{
    int octave = 0, layer = 0, best_octave = 0;
    float min_diff = FLT_MAX;
    for( octave = 1; octave < params.nOctaves; octave++ )
        for( layer = 0; layer < params.nOctaveLayers; layer++ ) {
            float diff = std::abs(kpt.size - (float)((HAAR_SIZE0 + HAAR_SIZE_INC*layer) << octave));
            if( min_diff > diff ) {
                min_diff = diff;
                best_octave = octave;
                if( min_diff == 0 )
                    return best_octave;
            }
        }
    return best_octave;
}


void FoveatedSURF::operator()(const Mat& img, const Mat& mask,
                      vector<KeyPoint>& keypoints) const
{
    CvMat _img = img, _mask, *pmask = 0;
    if( mask.data )
        pmask = &(_mask = mask);
    MemStorage storage(cvCreateMemStorage(0));
    Seq<CvSURFPoint> kp;
    cvExtractFoveatedSURF(&_img, pmask, &kp.seq, 0, storage, *(const CvFoveatedSURFParams*)this, 0);
    Seq<CvSURFPoint>::iterator it = kp.begin();
    size_t i, n = kp.size();
    keypoints.resize(n);
    for( i = 0; i < n; i++, ++it )
    {
        const CvSURFPoint& kpt = *it;
        keypoints[i] = KeyPoint(kpt.pt, (float)kpt.size, kpt.dir, kpt.hessian, getPointOctave(kpt, *this));
    }
}

void FoveatedSURF::operator()(const Mat& img, const Mat& mask,
                vector<KeyPoint>& keypoints,
                vector<float>& descriptors,
                bool useProvidedKeypoints) const
{
    CvMat _img = img, _mask, *pmask = 0;
    if( mask.data )
        pmask = &(_mask = mask);
    MemStorage storage(cvCreateMemStorage(0));
    Seq<CvSURFPoint> kp;
    CvSeq* d = 0;
    size_t i, n;
    if( useProvidedKeypoints )
    {
        kp = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSURFPoint), storage);
        n = keypoints.size();
        for( i = 0; i < n; i++ )
        {
            const KeyPoint& kpt = keypoints[i];
            kp.push_back(cvSURFPoint(kpt.pt, 1, cvRound(kpt.size), kpt.angle, kpt.response));
        }
    }

    cvExtractFoveatedSURF(&_img, pmask, &kp.seq, &d, storage,
        *(const CvFoveatedSURFParams*)this, useProvidedKeypoints);

    // input keypoints can be filtered in cvExtractFoveatedSURF()
    if( !useProvidedKeypoints || (useProvidedKeypoints && keypoints.size() != kp.size()) )
    {
        Seq<CvSURFPoint>::iterator it = kp.begin();
        size_t i, n = kp.size();
        keypoints.resize(n);
        for( i = 0; i < n; i++, ++it )
        {
            const CvSURFPoint& kpt = *it;
            keypoints[i] = KeyPoint(kpt.pt, (float)kpt.size, kpt.dir,
                                    kpt.hessian, getPointOctave(kpt, *this),
                                    kpt.laplacian);
        }
    }
    descriptors.resize(d ? d->total*d->elem_size/sizeof(float) : 0);
    if(descriptors.size() != 0)
        cvCvtSeqToArray(d, &descriptors[0]);
}

}
