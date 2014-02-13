#ifndef LINEAR_FOVEATION
#define LINEAR_FOVEATION

int getDelta(int k, int S0, int Sm, int m, int f) {
	return (k*(S0 - Sm + 2*f))/(2*m);
}

int getSize(int k, int S0, int Sm, int m) {
	return (k*Sm - k*S0 + m*S0)/m;
}

#endif

