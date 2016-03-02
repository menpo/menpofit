#include <math.h>
#include <string.h>

void *process(double *A, double *B, double *C, int r_a, int c_a, int r_b, int c_b, int r_c, int c_c, int num_feat) {
	for (int f = 0; f < num_feat; f++) {
		double *dst = C;
		double *A_src = A + f*r_a*c_a;      
		double *B_src = B + f*r_b*c_b;
		for (int y = 0; y < r_c; y++) {
			for (int x = 0; x < c_c; x++) {
				double val = 0;
				for (int yp = 0; yp < r_b; yp++) {
					  double *A_off = A_src + (y+yp)*c_a + x;
					  double *B_off = B_src + yp*c_b;
					  for (int xp = 0; xp < c_b; xp++) {
						  val += *(A_off++) * *(B_off++);
						}
				}
				*(dst++) += val;
			  }
		}
  }	
}

