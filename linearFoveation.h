
/*
   Copyright (C) 2014, Rafael Beserra <rafaelufrn@gmail.com>
   If you use this software for academic purposes, consider citing the related paper: Rafael Beserra Gomes, Bruno Motta de Carvalho, Luiz Marcos Garcia Gonçalves, Visual attention guided features selection with foveated images, Neurocomputing, Volume 120, 23 November 2013, Pages 34-44, ISSN 0925-2312, http://dx.doi.org/10.1016/j.neucom.2012.10.033.

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
 */


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
		fx = MIN((ux - wx)/2 - growthfactor, fx);
		fx = MAX((wx - ux)/2 + growthfactor, fx);
		fy = MIN((uy - wy)/2 - growthfactor, fy);
		fy = MAX((wy - uy)/2 + growthfactor, fy);
	}

	void setFovea(int imgX, int imgY) {
		fx = imgX - ux/2;
		fy = imgY - uy/2;
		fixFovea();
	}

	void check() {
		assert(wx > 0 && wx < ux);
		assert(wy > 0 && wy < uy);
		assert(ux > 0 && uy > 0);
		assert(m >= 1);
		assert(beta.size() == eta.size());
		assert(eta.size() == level.size());
		for(unsigned int i = 0; i < beta.size(); i++) {
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

