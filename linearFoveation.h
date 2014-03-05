#ifndef LINEAR_FOVEATION
#define LINEAR_FOVEATION

struct LinearFoveation {

	LinearFoveation() {
		fx = fy = growthfactor = 0;
		wx = wy = ux = uy = 0;
		m = 0;
	}

	inline int getDeltax(int k) {
		return (k*(ux - wx + 2*fx))/(2*m);
	}
	inline int getDeltay(int k) {
		return (k*(uy - wy + 2*fy))/(2*m);
	}

	inline int getSizex(int k) {
		return (k*wx - k*ux + m*ux)/m;
	}
	inline int getSizey(int k) {
		return (k*wy - k*uy + m*uy)/m;
	}

	//fix the fovea position: if fovea is outsite image domain, snap it to the closest valid position independently for each coordinate
	inline void fixFovea() {
		fx = MIN((ux - wx)/2, fx);
		fx = MAX((wx - ux)/2, fx);
		fy = MIN((uy - wy)/2, fy);
		fy = MAX((wy - uy)/2, fy);
	}

	void check() {
		assert(wx > 0 && wx < ux);
		assert(wy > 0 && wy < uy);
		assert(ux > 0 && uy > 0);
		assert(m >= 1);
		assert(beta.size() == eta.size());
		assert(eta.size() == level.size());
		for(int i = 0; i < beta.size(); i++) {
			assert(beta[i] == 1 || beta[i] == 0);
			assert(eta[i] >= 1);
			assert(level[i] >= 0 && level[i] <= m);
		}
		assert(growthfactor >= 0);
	}

	int wx, wy; //smallest level size
	int ux, uy; //image size
	int m; //numberOfLevels - 1
	int fx, fy; //fovea position
	int growthfactor;
	std::vector<int> beta;
	std::vector<int> eta;
	std::vector<int> level;
};


#endif

