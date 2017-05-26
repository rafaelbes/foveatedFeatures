#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {

	int ux, uy, uz, wx, wy, wz, levels, fx, fy, fz, scale;
	int deltax, deltay, deltaz, skx, sky, skz, m;

	if(argc <= 10) {
		fprintf(stderr, "Please provide these command line arguments (integers):\n");
		fprintf(stderr, "ux uy uz wx wy wz levels fx fy fz\n");
		return 1;
	}

	ux = atoi(argv[1]);
	uy = atoi(argv[2]);
	uz = atoi(argv[3]);

	wx = atoi(argv[4]);
	wy = atoi(argv[5]);
	wz = atoi(argv[6]);

	levels = atoi(argv[7]);
	fx = atoi(argv[8]);
	fy = atoi(argv[9]);
	fz = atoi(argv[10]);

	scale = 2;
	ux /= scale;
	uy /= scale;
	uz /= scale;
	wx /= scale;
	wy /= scale;
	wz /= scale;
	fx /= scale;
	fy /= scale;
	fz /= scale;

	m = levels-1;

	printf("\\documentclass[dvips]{article}\n");
	printf("		\\usepackage{pstricks-add}\n");
	printf("		\\usepackage{pst-eps}\n");
	printf("		\\usepackage{pst-3dplot}\n");
	printf("		\\begin{document}\n");
	printf("		\\begin{TeXtoEPS}\n");
	printf("			\\begin{pspicture}(0,0)(%d, %d)\n", ux, uy);
	printf("			\\psset{unit=0.1cm}\n");
	printf("			\\pstThreeDCoor[linewidth=3pt,xMax=30,yMax=30,zMax=30]\n");

//	printf("				\\psgrid[gridcolor=black!10,subgridcolor=black!10,gridlabels=0pt,subgriddiv=1](0,0)(%d,%d)\n", ux, uy);
	//printf("				\\pcline{|-|}(0,-4)(%d,-4)\n", ux);

	printf("				\\pstThreeDLine{|-|}(0,%d,0)(%d,%d,0)\n", uy+3,ux,uy+3);
	printf("\\pstThreeDPut(%d,%d,0){\\Huge $S_{0_x}$}", ux/2, uy+15);

	printf("				\\pstThreeDLine{|-|}(%d,%d,0)(%d,%d,%d)\n", -5,uy+5,-5,uy+5,uz);
	printf("\\pstThreeDPut(%d,%d,%d){\\Huge $S_{0_z}$}", 0, uy+18, uz/2);

	printf("				\\pstThreeDLine{|-|}(%d,%d,0)(%d,%d,0)\n", ux+3,0,ux+3,uy);
	printf("\\pstThreeDPut(%d,%d,%d){\\Huge $S_{0_y}$}", ux+20, uy/2, 0);


	//fovea
//	printf("				\\pcline{o->}(%d,%d)(%d,%d)\n", ux/2, uy/2, ux/2 + fx, uy/2 + fy);
//	printf("	\\naput[labelsep=1.4]{\\huge $F$}");


	for(int k = 0; k < levels; k++) {
		deltax = (k*(ux - wx + 2*fx))/(2*m);
		deltay = (k*(uy - wy + 2*fy))/(2*m);
		deltaz = (k*(uz - wz + 2*fz))/(2*m);
		skx = (k*wx - k*ux + m*ux)/m;
		sky = (k*wy - k*uy + m*uy)/m;
		skz = (k*wz - k*uz + m*uz)/m;
//		printf("\\psframe(%d, %d)(%d, %d)\n", deltax, deltay, deltax + skx, deltay + sky);
		printf("\\pstThreeDBox(%d,%d,%d)(0,0,%d)(%d,0,0)(0,%d,0)", deltax, deltay, deltaz, skz, skx, sky);
	
		//delta
		int sp = 18;
		int bsp = 1;
		if(k != 0) {
			printf("				\\pstThreeDLine{|-|}(%d,%d,%d)(%d,%d,%d)\n", 0, uy+bsp+k*sp, 0, deltax, uy+bsp+k*sp, 0);
//			printf("	\\uput[0](%d,%d){\\huge $\\delta x_%d$}", ux/2, -uy/3+deltay, k);
			printf("				\\pstThreeDLine[linestyle=dotted]{-}(%d,%d,%d)(%d,%d,%d)\n", deltax, uy+bsp+k*sp, 0, deltax, deltay+sky, 0);
			printf("				\\pstThreeDLine[linestyle=dotted]{-}(%d,%d,%d)(%d,%d,%d)\n", deltax, deltay+sky, deltaz, deltax, deltay+sky, 0);
			printf("\\pstThreeDPut(%d,%d,0){\\huge $\\delta x_%d$}", deltax-deltax/2-3, -9+uy+bsp+k*sp, k);

/*			printf("				\\pcline{|-|}(%d,%d)(%d,%d)\n", 0, -5-k*8, deltax, -5-k*8);
			printf("	\\nbput[labelsep=0.4]{\\huge $\\delta x_%d$}", k);
			printf("				\\pcline[linestyle=dotted]{-}(%d,%d)(%d,%d)\n", deltax, -5-k*8, deltax, deltay);
			printf("				\\pcline{|-|}(%d,%d)(%d,%d)\n", deltax, deltay-4, deltax + skx, deltay-4);
			printf("	\\nbput[labelsep=0.4]{\\huge $S_{%d_x}$}", k);
			
			printf("				\\pcline{|-|}(%d,%d)(%d,%d)\n", -3-k*12, 0, -3-k*12, deltay);
			printf("	\\naput[labelsep=0.4]{\\huge $\\delta y_%d$}", k);
			printf("				\\pcline[linestyle=dotted]{-}(%d,%d)(%d,%d)\n", -3-k*12, deltay, deltax, deltay);
			printf("				\\pcline{|-|}(%d,%d)(%d,%d)\n", deltax - 4, deltay, deltax - 4, deltay + sky);
			printf("	\\naput[labelsep=0.4]{\\huge $S_{%d_y}$}", k);*/
		}
	}
	
	printf("			\\end{pspicture}\n");
	printf("		\\end{TeXtoEPS}\n");
	printf("		\\end{document}\n");


	return 0;

}

