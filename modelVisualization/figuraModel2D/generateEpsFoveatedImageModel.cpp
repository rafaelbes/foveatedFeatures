#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX(A, B) ((A) > (B) ? (A) : (B))
#define MIN(A, B) ((A) < (B) ? (A) : (B))

int main(int argc, char **argv) {

	if(argc <= 7) {
		fprintf(stderr, "Please provide these command line arguments (integers):\n");
		fprintf(stderr, "ux uy wx wy levels fx fy\n");
		return 1;
	}

	int ux = atoi(argv[1]);
	int uy = atoi(argv[2]);
	int wx = atoi(argv[3]);
	int wy = atoi(argv[4]);
	int levels = atoi(argv[5]);
	int fx = atoi(argv[6]);
	int fy = atoi(argv[7]);
	int scale = 2;
	ux /= scale;
	uy /= scale;
	wx /= scale;
	wy /= scale;
	fx /= scale;
	fy /= scale;

	int deltax, deltay;
	int skx, sky;
	int m = levels-1;

	printf("\\documentclass[dvips]{article}\n");
	printf("		\\usepackage{pstricks-add}\n");
	printf("		\\usepackage{pst-eps}\n");
	printf("		\\begin{document}\n");
	printf("		\\begin{TeXtoEPS}\n");
	printf("			\\begin{pspicture}(%d, %d)\n", ux, uy);
	printf("			\\psset{unit=0.1cm}\n", ux, uy);

//	printf("				\\psgrid[gridcolor=black!10,subgridcolor=black!10,gridlabels=0pt,subgriddiv=1](0,0)(%d,%d)\n", ux, uy);
	printf("				\\pcline{|-|}(0,-4)(%d,-4)\n", ux);
	printf("	\\nbput[labelsep=1.4]{\\huge $S_{0,x}$}");
	printf("				\\pcline{|-|}(-4,0)(-4,%d)\n", uy);
	printf("	\\naput[labelsep=1.4]{\\huge $S_{0,y}$}");

	//fovea
	printf("				\\pcline{o->}(%d,%d)(%d,%d)\n", ux/2, uy/2, ux/2 + fx, uy/2 + fy);
	printf("	\\naput[labelsep=1.4]{\\huge $F$}");


	for(int k = 0; k < levels; k++) {
		deltax = (k*(ux - wx + 2*fx))/(2*m);
		deltay = (k*(uy - wy + 2*fy))/(2*m);
		skx = (k*wx - k*ux + m*ux)/m;
		sky = (k*wy - k*uy + m*uy)/m;
		printf("\\psframe(%d, %d)(%d, %d)\n", deltax, deltay, deltax + skx, deltay + sky);
		printf("\\psframe[linestyle=dashed](%d, %d)(%d, %d)\n", MAX(0,deltax), MAX(0,deltay), MIN(ux, deltax + skx), MIN(uy, deltay + sky));
	
		//delta
		if(k != 0) {
			printf("				\\pcline{|-|}(%d,%d)(%d,%d)\n", 0, -5-k*8, deltax, -5-k*8);
			printf("	\\nbput[labelsep=0.4]{\\huge $\\delta x_%d$}", k);
			printf("				\\pcline[linestyle=dotted]{-}(%d,%d)(%d,%d)\n", deltax, -5-k*8, deltax, deltay);
			printf("				\\pcline{|-|}(%d,%d)(%d,%d)\n", deltax, deltay-4, deltax + skx, deltay-4);
			printf("	\\nbput[labelsep=0.4]{\\huge $S_{%d,x}$}", k);
			
			printf("				\\pcline{|-|}(%d,%d)(%d,%d)\n", -3-k*12, 0, -3-k*12, deltay);
			printf("	\\naput[labelsep=0.4]{\\huge $\\delta y_%d$}", k);
			printf("				\\pcline[linestyle=dotted]{-}(%d,%d)(%d,%d)\n", -3-k*12, deltay, deltax, deltay);
			printf("				\\pcline{|-|}(%d,%d)(%d,%d)\n", deltax - 4, deltay, deltax - 4, deltay + sky);
			printf("	\\naput[labelsep=0.4]{\\huge $S_{%d,y}$}", k);
		}
	}
	
	printf("			\\end{pspicture}\n");
	printf("		\\end{TeXtoEPS}\n");
	printf("		\\end{document}\n");


	return 0;

}

