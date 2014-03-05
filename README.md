foveatedFeatures
================

This is the source code of the Foveated Features Extraction, as a result of research done by Rafael Beserra.
It uses Opencv2.4.8 for implementation and no more others dependencies.
Consider citing the related paper to this project:
Rafael Beserra Gomes, Bruno Motta de Carvalho, Luiz Marcos Garcia Gonçalves, Visual attention guided features selection with foveated images, Neurocomputing, Volume 120, 23 November 2013, Pages 34-44, ISSN 0925-2312, http://dx.doi.org/10.1016/j.neucom.2012.10.033.

Usage
----------------
First, you must create a yml file that contains initial values of the foveated model. You MUST specify the following parameters:
- smallestLevelWidth: 0 < value < imageWidth
- smallestLevelHeight: 0 < value < imageHeight
- numberOfLevels: 
- bvector
- etavector
- levelvector: a 

Second, use the struct FoveatedHessianDetectorParams to specify:
- the image size (no default value)
- the YML file path containing the foveated model parameters (no default file)
- hessian threshold (default value: 100);
- the number of layers in each octave (it is preferable to use default value: 3).
Use FoveatedHessianDetectorParams(int imageWidth, int imageHeight, String ymlFile) construtor to specify the original image size and the yml file file path.

After that, use foveatedHessianDetector function:
static void foveatedHessianDetector(InputArray _img, InputArray _mask, vector<KeyPoint>& keypoints, FoveatedHessianDetectorParams params);
