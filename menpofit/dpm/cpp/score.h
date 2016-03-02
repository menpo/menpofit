#include <math.h>
#include <stdint.h>

// score(w,qp.x,inds)
// scores a weight vector 'w' on a set of sparse examples in qp.x at the columns specified by 'inds'
static inline double cscore(const double *W, const float* x) {
  double y  = 0;
  int    xp = 1;
  // Iterate through blocks, and grab boundary indices using matlab's indexing
  for (int b = 0; b < x[0]; b++) {
    int wp  = (int)x[xp++];
    int len = (int)x[xp++] - wp;
    for (int i = 0; i < len; i++) {
      y += W[wp++] * (double)x[xp++];
    }
  }
  return y;
}