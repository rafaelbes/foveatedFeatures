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
//#include "precomp.hpp"

#include "foveatedSurf.h"

namespace cv
{
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


	static void foveatedFastHessianDetector( const Mat& sum, const Mat& mask_sum, vector<KeyPoint>& keypoints, int nOctaves, int nOctaveLayers, float hessianThreshold ) {
		/* Sampling step along image x and y axes at first octave. This is doubled
		   for each additional octave. WARNING: Increasing this improves speed,
		   however keypoint extraction becomes unreliable. */
		const int SAMPLE_STEP0 = 1;

		int nTotalLayers = (nOctaveLayers+2)*nOctaves;
		int nMiddleLayers = nOctaveLayers*nOctaves;

		vector<Mat> dets(nTotalLayers);
		vector<Mat> traces(nTotalLayers);
		vector<int> sizes(nTotalLayers);
		vector<int> sampleSteps(nTotalLayers);
		vector<int> middleIndices(nMiddleLayers);

		// Allocate space and calculate properties of each layer
		int index = 0, middleIndex = 0, step = SAMPLE_STEP0;

		for( int octave = 0; octave < nOctaves; octave++ )
		{
			for( int layer = 0; layer < nOctaveLayers+2; layer++ )
			{
				/* The integral image sum is one pixel bigger than the source image*/
				dets[index].create( (sum.rows-1)/step, (sum.cols-1)/step, CV_32F );
				traces[index].create( (sum.rows-1)/step, (sum.cols-1)/step, CV_32F );
				sizes[index] = (SURF_HAAR_SIZE0 + SURF_HAAR_SIZE_INC*layer) << octave;
				sampleSteps[index] = step;

				if( 0 < layer && layer <= nOctaveLayers )
					middleIndices[middleIndex++] = index;
				index++;
			}
			step *= 2;
		}

		// Calculate hessian determinant and trace samples in each layer
		parallel_for( BlockedRange(0, nTotalLayers),
				SURFBuildInvoker(sum, sizes, sampleSteps, dets, traces) );

		// Find maxima in the determinant of the hessian
		parallel_for( BlockedRange(0, nMiddleLayers),
				SURFFindInvoker(sum, mask_sum, dets, traces, sizes,
					sampleSteps, middleIndices, keypoints,
					nOctaveLayers, hessianThreshold) );

		std::sort(keypoints.begin(), keypoints.end(), KeypointGreater());
	}



} //end namespace
