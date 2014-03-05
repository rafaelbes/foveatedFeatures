foveatedFeatures
================

This is the source code of the Foveated Features Extraction, as a result of research done by Rafael Beserra.
It uses Opencv2.4.8 for implementation and no more others dependencies.
Consider citing the related paper to this project:
Rafael Beserra Gomes, Bruno Motta de Carvalho, Luiz Marcos Garcia Gonçalves, Visual attention guided features selection with foveated images, Neurocomputing, Volume 120, 23 November 2013, Pages 34-44, ISSN 0925-2312, http://dx.doi.org/10.1016/j.neucom.2012.10.033.

Foveation variable summary
----------------
The feature extraction process is done by a sequence of n passes. This differs slight from the original paper. If you want a easy configuration, use one of these values:

- numberOfLevels = 4, etavector = [4, 3, 2, 1], levels = [0, 1, 2, 3], b = [1, 1, 1, 1]
- numberOfLevels = 5, etavector = [5, 4, 3, 2, 1], levels = [0, 1, 2, 3, 4], b = [1, 1, 1, 1]

In the original paper, there are a B vector and a eta vector, each one with numberOfLevels elements. For example, 4 levels, B = {1, 1, 0, 1} and eta = {4, 3, 2, 1}, this set means that there are 4 passes for feature extraction:
- first level (largest one) computes features in the fourth octave
- second level computes features in the third octave
- third level would compute features in the second octave, but is does not because B[3] = 0
- fourth level computes features in the first octave

See section 3.2 from neurocomputing paper for examples.

In this implementation, bvector (B), etavector (eta) and levelvector have n elements, the set {bvector[i], etavector[i], levelvector[i]} represents a feature extraction pass.
- bvector: [b1, b2, ..., bn]: a vector where bi is 0 if the feature extraction pass number i should be discarded or 1, otherwise
- etavector: [e1, e2, ..., en]: a vector where ei is the octave (> 0) for which the feature extraction pass number i should be performed
- levelvector: [l1, l2, ..., ln]: a vector where li is the foveated model level (>= 0 and < numberOfLevels) for which the feature extraction pass number i should be performed

For example: 4 levels, B = {1, 0, 1, 1, 1}, eta = {3, 4, 2, 3, 1} and levels = {0, 0, 1, 1, 3}, this set means that there are 5 passes for feature extraction:
- first pass (B[1] = 1, eta[1] = 3 and levels[1] = 0): a feature extraction in the third octave is performed in the first level (largest one)
- second pass (B[2] = 0, eta[2] = 4 and levels[2] = 0): a feature extraction in the fourth octave would be performed in the first level (largest one), but it does not because B[2] = 0
- third pass (B[3] = 1, eta[3] = 2 and levels[3] = 1): a feature extraction in the second octave is performed in the second level
- fourth pass (B[4] = 1, eta[4] = 3 and levels[4] = 1): a feature extraction in the third octave is performed in the second level
- fifth pass (B[5] = 1, eta[5] = 1 and levels[5] = 3): a feature extraction in the first octave is performed in the fourth level

This also means that:
- first level (largest one) computes features in the third octave (levels[1] = 0, levels[2] = 0, eta[1] = 3, eta[2] = 4, but B[2] = 0)
- second level computes features in the second and third octave (levels[3] = 1, levels[4] = 1, eta[3] = 2, eta[4] = 3)
- third level has no feature extraction (since no level = 2)
- fourth level computes features in the first octave (levels[5] = 3, eta[5] = 1)

Usage
----------------
First, you must create a yml file that contains initial values of the foveated model. You MUST specify the following parameters:
- smallestLevelWidth: 0 < value < imageWidth
- smallestLevelHeight: 0 < value < imageHeight
- numberOfLevels: value > 1
- bvector: [b1, b2, ..., bn], where each value is 0 or 1
- etavector: [e1, e2, ..., en], where each value > 0
- levelvector: [l1, l2, ..., ln], where value >= 0 and value < numberOfLevels
Note that these last 3 sequence must have the same number of elements

Second, use the struct FoveatedHessianDetectorParams to specify:
- the image size (no default value)
- the YML file path containing the foveated model parameters (no default file)
- hessian threshold (default value: 100);
- the number of layers in each octave (it is preferable to use default value: 3).
Use FoveatedHessianDetectorParams(int imageWidth, int imageHeight, String ymlFile) construtor to specify the original image size and the yml file file path.

After that, use foveatedHessianDetector function:
static void foveatedHessianDetector(InputArray _img, InputArray _mask, vector<KeyPoint>& keypoints, FoveatedHessianDetectorParams params);
