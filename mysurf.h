#ifndef MYSURF_H
#define MYSURF_H

#include <cv.h>

struct myCvSurfHF {
    int p0, p1, p2, p3; //em coordenadas de vetor unidimensional
    float w;
};

//de acordo com a documentacao do opencv, são todos os pontos inclusives entre (p0, p3)
CV_INLINE float
foveatedCalcHaarPattern( const int* origin, const myCvSurfHF* f, int n )
{
	float d = 0;
    for( int k = 0; k < n; k++ )
        d += (origin[f[k].p0] + origin[f[k].p3] - origin[f[k].p1] - origin[f[k].p2])*f[k].w;
    return d;
}

/*
	* (p0)----(p2)
	*  |       |
	* (p1)----(p3)
   */
static void
foveatedResizeHaarPattern( const int src[][5], myCvSurfHF* dst, int n, int oldSize, int newSize, int widthStep )
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

namespace cv
{

struct SURFInvoker
{
    enum { ORI_RADIUS = 6, ORI_WIN = 60, PATCH_SZ = 20 };
    
    static const int ORI_SEARCH_INC;
    static const float ORI_SIGMA;
    static const float DESC_SIGMA;
    
    SURFInvoker( const CvSURFParams* _params,
                 CvSeq* _keypoints, CvSeq* _descriptors,
                 const CvMat* _img, const CvMat* _sum, 
                 const CvPoint* _apt, const float* _aptw,
                 int _nangle0, const float* _DW )
    {
        params = _params;
        keypoints = _keypoints;
        descriptors = _descriptors;
        img = _img;
        sum = _sum;
        apt = _apt;
        aptw = _aptw;
        nangle0 = _nangle0;
        DW = _DW;
    }
    
    void operator()(const BlockedRange& range) const
    {
        /* X and Y gradient wavelet data */
        const int NX=2, NY=2;
        int dx_s[NX][5] = {{0, 0, 2, 4, -1}, {2, 0, 4, 4, 1}};
        int dy_s[NY][5] = {{0, 0, 4, 2, 1}, {0, 2, 4, 4, -1}};
        
        const int descriptor_size = params->extended ? 128 : 64;
        
        const int max_ori_samples = (2*ORI_RADIUS+1)*(2*ORI_RADIUS+1);
        float X[max_ori_samples], Y[max_ori_samples], angle[max_ori_samples];
        uchar PATCH[PATCH_SZ+1][PATCH_SZ+1];
        float DX[PATCH_SZ][PATCH_SZ], DY[PATCH_SZ][PATCH_SZ];
        
        CvMat matX = cvMat(1, max_ori_samples, CV_32F, X);
        CvMat matY = cvMat(1, max_ori_samples, CV_32F, Y);
        CvMat _angle = cvMat(1, max_ori_samples, CV_32F, angle);
        CvMat _patch = cvMat(PATCH_SZ+1, PATCH_SZ+1, CV_8U, PATCH);
        
        int k, k1 = range.begin(), k2 = range.end();
        int maxSize = 0;
        
        for( k = k1; k < k2; k++ )
            maxSize = std::max(maxSize, ((CvSURFPoint*)cvGetSeqElem( keypoints, k ))->size);
        
        maxSize = cvCeil((PATCH_SZ+1)*maxSize*1.2f/9.0f);
		if(maxSize <= 0) return;

        Ptr<CvMat> winbuf = cvCreateMat( 1, maxSize*maxSize, CV_8U );
        
        for( k = k1; k < k2; k++ )
        {
            const int* sum_ptr = sum->data.i;
            int sum_cols = sum->cols;
            int i, j, kk, x, y, nangle;
            
            float* vec;
            myCvSurfHF dx_t[NX], dy_t[NY];
            
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
            foveatedResizeHaarPattern( dx_s, dx_t, NX, 4, grad_wav_size, sum->cols );
            foveatedResizeHaarPattern( dy_s, dy_t, NY, 4, grad_wav_size, sum->cols );
            for( kk = 0, nangle = 0; kk < nangle0; kk++ )
            {
                const int* ptr;
                float vx, vy;
                x = cvRound( center.x + apt[kk].x*s - (float)(grad_wav_size-1)/2 );
                y = cvRound( center.y + apt[kk].y*s - (float)(grad_wav_size-1)/2 );
                if( (unsigned)y >= (unsigned)(sum->rows - grad_wav_size) ||
                   (unsigned)x >= (unsigned)(sum->cols - grad_wav_size) )
                    continue;
                ptr = sum_ptr + x + y*sum_cols;
                vx = foveatedCalcHaarPattern( ptr, dx_t, 2 );
                vy = foveatedCalcHaarPattern( ptr, dy_t, 2 );
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
            
            float descriptor_dir = cvFastArctan( besty, bestx );
            kp->dir = descriptor_dir;
            
            if( !descriptors )
                continue;
            
            descriptor_dir *= (float)(CV_PI/180);
            
            /* Extract a window of pixels around the keypoint of size 20s */
            int win_size = (int)((PATCH_SZ+1)*s);
            CV_Assert( winbuf->cols >= win_size*win_size );
            
            CvMat win = cvMat(win_size, win_size, CV_8U, winbuf->data.ptr);
            float sin_dir = sin(descriptor_dir);
            float cos_dir = cos(descriptor_dir) ;
            
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
            uchar* WIN = win.data.ptr;
            for( i = 0; i < win_size; i++, start_x += sin_dir, start_y += cos_dir )
            {
                float pixel_x = start_x;
                float pixel_y = start_y;
                for( j = 0; j < win_size; j++, pixel_x += cos_dir, pixel_y -= sin_dir )
                {
                    int x = std::min(std::max(cvRound(pixel_x), 0), img->cols-1);
                    int y = std::min(std::max(cvRound(pixel_y), 0), img->rows-1);
                    WIN[i*win_size + j] = img->data.ptr[y*img->step + x];
                }
            }
            
            /* Scale the window to size PATCH_SZ so each pixel's size is s. This
             makes calculating the gradients with wavelets of size 2s easy */
            cvResize( &win, &_patch, CV_INTER_AREA );
            
            /* Calculate gradients in x and y with wavelets of size 2s */
            for( i = 0; i < PATCH_SZ; i++ )
                for( j = 0; j < PATCH_SZ; j++ )
                {
                    float dw = DW[i*PATCH_SZ + j];
                    float vx = (PATCH[i][j+1] - PATCH[i][j] + PATCH[i+1][j+1] - PATCH[i+1][j])*dw;
                    float vy = (PATCH[i+1][j] - PATCH[i][j] + PATCH[i+1][j+1] - PATCH[i][j+1])*dw;
                    DX[i][j] = vx;
                    DY[i][j] = vy;
                }
            
            /* Construct the descriptor */
            vec = (float*)cvGetSeqElem( descriptors, k );
            for( kk = 0; kk < (int)(descriptors->elem_size/sizeof(vec[0])); kk++ )
                vec[kk] = 0;
            double square_mag = 0;       
            if( params->extended )
            {
                /* 128-bin descriptor */
                for( i = 0; i < 4; i++ )
                    for( j = 0; j < 4; j++ )
                    {
                        for( y = i*5; y < i*5+5; y++ )
                        {
                            for( x = j*5; x < j*5+5; x++ )
                            {
                                float tx = DX[y][x], ty = DY[y][x];
                                if( ty >= 0 )
                                {
                                    vec[0] += tx;
                                    vec[1] += (float)fabs(tx);
                                } else {
                                    vec[2] += tx;
                                    vec[3] += (float)fabs(tx);
                                }
                                if ( tx >= 0 )
                                {
                                    vec[4] += ty;
                                    vec[5] += (float)fabs(ty);
                                } else {
                                    vec[6] += ty;
                                    vec[7] += (float)fabs(ty);
                                }
                            }
                        }
                        for( kk = 0; kk < 8; kk++ )
                            square_mag += vec[kk]*vec[kk];
                        vec += 8;
                    }
            }
            else
            {
                /* 64-bin descriptor */
                for( i = 0; i < 4; i++ )
                    for( j = 0; j < 4; j++ )
                    {
                        for( y = i*5; y < i*5+5; y++ )
                        {
                            for( x = j*5; x < j*5+5; x++ )
                            {
                                float tx = DX[y][x], ty = DY[y][x];
                                vec[0] += tx; vec[1] += ty;
                                vec[2] += (float)fabs(tx); vec[3] += (float)fabs(ty);
                            }
                        }
                        for( kk = 0; kk < 4; kk++ )
                            square_mag += vec[kk]*vec[kk];
                        vec+=4;
                    }
            }
            
            /* unit vector is essential for contrast invariance */
            vec = (float*)cvGetSeqElem( descriptors, k );
            double scale = 1./(sqrt(square_mag) + DBL_EPSILON);
            for( kk = 0; kk < descriptor_size; kk++ )
                vec[kk] = (float)(vec[kk]*scale);
        }
    }
   
    const CvSURFParams* params;
    const CvMat* img;
    const CvMat* sum;
    CvSeq* keypoints;
    CvSeq* descriptors;
    const CvPoint* apt;
    const float* aptw;
    int nangle0;
    const float* DW;
};

const int SURFInvoker::ORI_SEARCH_INC = 5;  
const float SURFInvoker::ORI_SIGMA = 2.5f;
const float SURFInvoker::DESC_SIGMA = 3.3f;
    
}

// PRIMEIRA PARTE: DETECTOR HESSIANO
CV_INLINE int 
foveatedInterpolateKeypoint( float N9[3][9], int dx, int dy, int ds, CvSURFPoint *point )
{
    int solve_ok;
    float A[9], x[3], b[3];
    CvMat matA = cvMat(3, 3, CV_32F, A);
    CvMat _x = cvMat(3, 1, CV_32F, x);                
    CvMat _b = cvMat(3, 1, CV_32F, b);

    b[0] = -(N9[1][5]-N9[1][3])/2;  /* Negative 1st deriv with respect to x */
    b[1] = -(N9[1][7]-N9[1][1])/2;  /* Negative 1st deriv with respect to y */
    b[2] = -(N9[2][4]-N9[0][4])/2;  /* Negative 1st deriv with respect to s */

    A[0] = N9[1][3]-2*N9[1][4]+N9[1][5];            /* 2nd deriv x, x */
    A[1] = (N9[1][8]-N9[1][6]-N9[1][2]+N9[1][0])/4; /* 2nd deriv x, y */
    A[2] = (N9[2][5]-N9[2][3]-N9[0][5]+N9[0][3])/4; /* 2nd deriv x, s */
    A[3] = A[1];                                    /* 2nd deriv y, x */
    A[4] = N9[1][1]-2*N9[1][4]+N9[1][7];            /* 2nd deriv y, y */
    A[5] = (N9[2][7]-N9[2][1]-N9[0][7]+N9[0][1])/4; /* 2nd deriv y, s */
    A[6] = A[2];                                    /* 2nd deriv s, x */
    A[7] = A[5];                                    /* 2nd deriv s, y */
    A[8] = N9[0][4]-2*N9[1][4]+N9[2][4];            /* 2nd deriv s, s */

    solve_ok = cvSolve( &matA, &_b, &_x );
    if( solve_ok )
    {
        point->pt.x += x[0]*dx;
        point->pt.y += x[1]*dy;
        point->size = cvRound( point->size + x[2]*ds ); 
    }
    return solve_ok;
}

/* Wavelet size at first layer of first octave. */ 
const int MYHAAR_SIZE0 = 9;    

/* Wavelet size increment between layers. This should be an even number, 
 such that the wavelet sizes in an octave are either all even or all odd.
 This ensures that when looking for the neighbours of a sample, the layers
 above and below are aligned correctly. */
const int MYHAAR_SIZE_INC = 6;

int preSizes[4][4] = {{9,15,21,27},{15,27,39,51},{27,51,75,99},{51,51+48,51+48+48,51+48*3}};

CV_INLINE int getWaveletSize(int octave, int layer) {
	return preSizes[octave][layer];
}

CV_INLINE int getWaveletMargin(int octave, int layer) {
	return getWaveletSize(octave, layer)/2;
}

//dada a posicao em det, retorna a posicao da wavelet
CV_INLINE int getWaveletCenter(int x, int octave, int layer) {
	return (((x-getWaveletMargin(octave, layer)))<<octave) + getWaveletMargin(octave, layer);
}

#define FOVEATEDSURF_REGION_PRINT 1

//o vetor B indica quais niveis estao presentes e quais nao estao
//o vetor eta indica em qual oitava esta o nivel correspondente
//m indica o numero de niveis - 1... (m = 3, os niveis sao 0, 1, 2 e 3)
static CvSeq* foveatedFastHessianDetector( const CvMat* sum, const CvMat* mask_sum,
    CvMemStorage* storage, const CvSURFParams* params, int fx, int fy, int wx, int wy, int *B, int *eta, int m )
{
    CvSeq* points = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvSURFPoint), storage );

    /* Sampling step along image x and y axes at first octave. This is doubled
       for each additional octave. WARNING: Increasing this improves speed, 
       however keypoint extraction becomes unreliable. */
    const int SAMPLE_STEP0 = 1; 

    /* Wavelet Data */
    const int NX=3, NY=3, NXY=4, NM=1;
    const int dx_s[NX][5] = { {0, 2, 3, 7, 1}, {3, 2, 6, 7, -2}, {6, 2, 9, 7, 1} };
    const int dy_s[NY][5] = { {2, 0, 7, 3, 1}, {2, 3, 7, 6, -2}, {2, 6, 7, 9, 1} };
    const int dxy_s[NXY][5] = { {1, 1, 4, 4, 1}, {5, 1, 8, 4, -1}, {1, 5, 4, 8, -1}, {5, 5, 8, 8, 1} };
    const int dm[NM][5] = { {0, 0, 9, 9, 1} };
    myCvSurfHF Dx[NX], Dy[NY], Dxy[NXY], Dm;

	double tt;
    CvMat** dets = (CvMat**)cvStackAlloc((params->nOctaveLayers+2)*sizeof(dets[0]));
    CvMat** traces = (CvMat**)cvStackAlloc((params->nOctaveLayers+2)*sizeof(traces[0]));
    int *sizes = (int*)cvStackAlloc((params->nOctaveLayers+2)*sizeof(sizes[0]));

    float dx = 0, dy = 0, dxy = 0;
    int octave, layer, sampleStep, size, margin;
	int level;
    int rows, cols;
    int i, j, sum_i, sum_j;
    const int* s_ptr;
    float *det_ptr, *trace_ptr;

    for( layer = 0; layer <= params->nOctaveLayers+1; layer++ )
    {
        dets[layer]   = cvCreateMat( (sum->rows-1)/SAMPLE_STEP0, (sum->cols-1)/SAMPLE_STEP0, CV_32FC1 );
        traces[layer] = cvCreateMat( (sum->rows-1)/SAMPLE_STEP0, (sum->cols-1)/SAMPLE_STEP0, CV_32FC1 );
    }

	for(level = 0; level <= m; level++) {
    //for( octave = 0, sampleStep=SAMPLE_STEP0; octave < params->nOctaves; octave++, sampleStep*=2 ) {
        rows = (sum->rows-1)/sampleStep;
        cols = (sum->cols-1)/sampleStep;
		int deltax, deltay;
		int limit_x, limit_y;
		int skx, sky;
		//int m = 2; //agora é passado como parâmetro
		sampleStep = 1 << eta[level];
		octave = eta[level];
		int k = level;
		deltax = (k*(sum->cols-1 - wx + 2*fx))/(2*m);
		deltay = (k*(sum->rows-1 - wy + 2*fy))/(2*m);

		int margin_x, margin_y;
		skx = (k*wx - k*(sum->cols-1) + m*(sum->cols-1))/m;
		sky = (k*wy - k*(sum->rows-1) + m*(sum->rows-1))/m;

		//todas as wavelets vão partir do mesmo ponto, que é determinado pela maior wavelet
		margin = getWaveletMargin(octave,params->nOctaveLayers+1);
		//e também elas devem começar a partir de (deltax, deltay), portanto, pega-se a maior coordenada
		int foveafactor = 0;
		margin_x = MAX(margin, deltax - foveafactor);
		margin_y = MAX(margin, deltay - foveafactor);
		//as wavelets vao ate um certo limite e tambem nao pode passar pela maior wavelet, portanto, pega-se a menor coordenada
		limit_x = MIN(deltax + skx + foveafactor, sum->cols-1 - margin);
		limit_y = MIN(deltay + sky + foveafactor, sum->rows-1 - margin);

		//printf("--------------------------------\n");
        for( layer = 0; layer <= params->nOctaveLayers+1; layer++ )
        {
			sizes[layer] = size = getWaveletSize(octave, layer);
            foveatedResizeHaarPattern( dx_s, Dx, NX, 9, size, sum->cols );
            foveatedResizeHaarPattern( dy_s, Dy, NY, 9, size, sum->cols );
            foveatedResizeHaarPattern( dxy_s, Dxy, NXY, 9, size, sum->cols );
           
            //printf( "octave=%d layer=%d size=%d rows=%d cols=%d, margin=%d\n", octave, layer, size, rows, cols, margin );
			//sum_i (y) e sum_j (em x) varrem a imagem de soma
			margin = getWaveletMargin(octave, layer);
			sum_i = margin_y - margin;
			float d = 0;
			tt = (double)cvGetTickCount();
			int ncols = dets[layer]->cols;
            for(i = 0; sum_i<=limit_y - margin; sum_i+=sampleStep, i++ )
            {
				sum_j = margin_x - margin;
                s_ptr = sum->data.i + sum_i*sum->cols + sum_j;
                det_ptr = dets[layer]->data.fl + i*ncols;
                trace_ptr = traces[layer]->data.fl + i*ncols;
                for(; sum_j<=limit_x - margin; sum_j+=sampleStep )
                {
                    dx  = foveatedCalcHaarPattern( s_ptr, Dx, 3 );
                    dy  = foveatedCalcHaarPattern( s_ptr, Dy, 3 );
                    dxy = foveatedCalcHaarPattern( s_ptr, Dxy, 4 );
                    s_ptr+=sampleStep;
					*det_ptr++ = (float)(dx*dy - 0.81*dxy*dxy);
					*trace_ptr++ = (float)(dx + dy);
                }
            }
		tt = (double)cvGetTickCount() - tt;
		//printf( "Extraction time = %gms\n", tt/(cvGetTickFrequency()*1000.));
        }
        /* Find maxima in the determinant of the hessian */
        for( layer = 1; layer <= params->nOctaveLayers; layer++ )
        {
			size = getWaveletSize(octave, layer);
            
			margin = getWaveletMargin(octave, params->nOctaveLayers+1);
			sum_i = margin_y;
            for( i = 0; sum_i <= limit_y; sum_i += sampleStep, i++ )
            {
				sum_j = margin_x;
                det_ptr = dets[layer]->data.fl + i*dets[layer]->cols;
                trace_ptr = traces[layer]->data.fl + i*traces[layer]->cols;
                for( j = 0; sum_j <= limit_x; sum_j += sampleStep, j++)
                {
					if(i == 0 || j == 0 || sum_i + sampleStep > limit_y || sum_j + sampleStep > limit_x) continue;

                    float val0 = det_ptr[j];
                    if( val0 > params->hessianThreshold )
                    {
                        int c = dets[layer]->cols;
						
                        const float *det1 = dets[layer-1]->data.fl + i*c + j;
						const float *det2 = dets[layer]->data.fl   + i*c + j;
                        const float *det3 = dets[layer+1]->data.fl + i*c + j;

                        float N9[3][9] = { { det1[-c-1], det1[-c], det1[-c+1],          
                                             det1[-1]  , det1[0] , det1[1],
                                             det1[c-1] , det1[c] , det1[c+1]  },
                                           { det2[-c-1], det2[-c], det2[-c+1],       
                                             det2[-1]  , det2[0] , det2[1],
                                             det2[c-1] , det2[c] , det2[c+1 ] },
                                           { det3[-c-1], det3[-c], det3[-c+1],       
                                             det3[-1  ], det3[0] , det3[1],
                                             det3[c-1] , det3[c] , det3[c+1 ] } };

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
                            double center_i = sum_i;
                            double center_j = sum_j;
							//center_i = (double) getWaveletCenter(i, octave, params->nOctaveLayers+1);
							//center_j = (double) getWaveletCenter(j, octave, params->nOctaveLayers+1);
							//printf("%f %f %d %d\n", center_i, center_j, i, j);

                            CvSURFPoint point = cvSURFPoint( cvPoint2D32f(center_j,center_i), 
                                                             CV_SIGN(trace_ptr[j]), sizes[layer], 0, val0 );
                           
                            /* Interpolate maxima location within the 3x3x3 neighbourhood  */
                            int ds = sizes[layer]-sizes[layer-1];
                            int interp_ok = foveatedInterpolateKeypoint( N9, sampleStep, sampleStep, ds, &point );

                            /* Sometimes the interpolation step gives a negative size etc. */
                            if( interp_ok && point.size >= 1 &&
                                point.pt.x >= 0 && point.pt.x <= (sum->cols-1) &&
                                point.pt.y >= 0 && point.pt.y <= (sum->rows-1) )
                            {    
                                /*printf( "KeyPoint %f %f %d\n", point.pt.x, point.pt.y, point.size );*/
                                cvSeqPush( points, &point );
                            }    
                        }
                    }
                }
            }
        }
    }

    /* Clean-up */
    for( layer = 0; layer <= params->nOctaveLayers+1; layer++ )
    {
        cvReleaseMat( &dets[layer] );
        cvReleaseMat( &traces[layer] );
    }

    return points;
}

// SEGUNDA PARTE
CV_IMPL void cvExtractFoveatedSURF(const CvArr* _img, const CvArr* _mask,
               CvSeq** _keypoints, CvSeq** _descriptors,
               CvMemStorage* storage, CvSURFParams params,
			   int useProvidedKeyPts, int fx, int fy, int wx, int wy, int *B, int *eta, int m ) {

    const int ORI_RADIUS = 6;
    const float ORI_SIGMA = 2.5f;
    const float DESC_SIGMA = 3.3f;
    
	CvMat *sum = 0, *mask1 = 0, *mask_sum = 0;

    if( _keypoints && !useProvidedKeyPts ) // If useProvidedKeyPts!=0 we'll use current contents of "*_keypoints"
        *_keypoints = 0;
    if( _descriptors )
        *_descriptors = 0;
    
	CvSeq *keypoints, *descriptors = 0;
    CvMat imghdr, *img = cvGetMat(_img, &imghdr);
    CvMat maskhdr, *mask = _mask ? cvGetMat(_mask, &maskhdr) : 0;
    
    const int max_ori_samples = (2*ORI_RADIUS+1)*(2*ORI_RADIUS+1);
    int descriptor_size = params.extended ? 128 : 64;
    const int descriptor_data_type = CV_32F;
    const int PATCH_SZ = 20;
    float DW[PATCH_SZ][PATCH_SZ];
    CvMat _DW = cvMat(PATCH_SZ, PATCH_SZ, CV_32F, DW);
    CvPoint apt[max_ori_samples];
    float aptw[max_ori_samples];
    int i, j, nangle0 = 0, N;

    CV_Assert(img != 0);
    CV_Assert(CV_MAT_TYPE(img->type) == CV_8UC1);
    CV_Assert(mask == 0 || (CV_ARE_SIZES_EQ(img,mask) && CV_MAT_TYPE(mask->type) == CV_8UC1));
    CV_Assert(storage != 0);
    CV_Assert(params.hessianThreshold >= 0);
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
		keypoints = foveatedFastHessianDetector( sum, mask_sum, storage, &params, fx, fy, wx, wy, B, eta, m);
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

    /* Coordinates and weights of samples used to calculate orientation */
    cv::Mat matG = cv::getGaussianKernel( 2*ORI_RADIUS+1, ORI_SIGMA, CV_32F );
    const float* G = (const float*)matG.data;
    
    for( i = -ORI_RADIUS; i <= ORI_RADIUS; i++ )
    {
        for( j = -ORI_RADIUS; j <= ORI_RADIUS; j++ )
        {
            if( i*i + j*j <= ORI_RADIUS*ORI_RADIUS )
            {
                apt[nangle0] = cvPoint(j,i);
                aptw[nangle0++] = G[i+ORI_RADIUS]*G[j+ORI_RADIUS];
            }
        }
    }

    /* Gaussian used to weight descriptor samples */
    double c2 = 1./(DESC_SIGMA*DESC_SIGMA*2);
    double gs = 0;
    for( i = 0; i < PATCH_SZ; i++ )
    {
        for( j = 0; j < PATCH_SZ; j++ )
        {
            double x = j - (float)(PATCH_SZ-1)/2, y = i - (float)(PATCH_SZ-1)/2;
            double val = exp(-(x*x+y*y)*c2);
            DW[i][j] = (float)val;
            gs += val;
        }
    }
    cvScale( &_DW, &_DW, 1./gs );

    cv::parallel_for(cv::BlockedRange(0, N),
                     cv::SURFInvoker(&params, keypoints, descriptors, img, sum,
                                     apt, aptw, nangle0, &DW[0][0]));
    //cv::SURFInvoker(&params, keypoints, descriptors, img, sum,
    //                apt, aptw, nangle0, &DW[0][0])(cv::BlockedRange(0, N));
   
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


CV_IMPL void cvStandardExtractFoveatedSURF(const CvArr* _img, const CvArr* _mask,
               CvSeq** _keypoints, CvSeq** _descriptors,
               CvMemStorage* storage, CvSURFParams params,
			   int useProvidedKeyPts, int fx, int fy, int wx, int wy, int m) {
	int *b = (int *) malloc(sizeof(int)*(m+1));
	int *eta = (int *) malloc(sizeof(int)*(m+1));
	for(int i = 0; i <= m; i++) {
		b[i] = 1;
		eta[i] = m - i + 1;
	}
	cvExtractFoveatedSURF(_img, _mask, _keypoints, _descriptors, storage, params, useProvidedKeyPts, fx, fy, wx, wy, b, eta, m);
	free(b);
	free(eta);
}


#endif

