#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <iostream>
using namespace std;

#define MAX(A,B) ((A) < (B) ? (B) : (A))
#define MIN(A,B) ((A) > (B) ? (B) : (A))
#define INDEX(X,I) (*(int *)(X + I*mm + m))
int n, m, mm;

// Comparison function for sorting function in sumAlpha function
static inline void setMN(int M, int N) {
    m = M;
    n = N;
}


int comp(const void *a, const void *b) {
  return memcmp((int32_t *)a,(int32_t *)b,m*sizeof(int32_t));
}

static inline double c_one_score(const double *W, const float* x) {
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

static inline double dot(const float *x, const float *y) {
  double res = 0;
  int xnum = (int)x[0];
  int ynum = (int)y[0];

  //  b: block number
  //  i: position in sparse vector
  //  j: position in dense vector
  // j1: start position in dense vector (matlab index)
  // j2: end   position in dense vector (matlab index)
  int yb=0, xb=0;
  int yi=1, xi=1;
  int yj1 = (int)y[yi++];
  int yj2 = (int)y[yi++];
  int xj1 = (int)x[xi++];
  int xj2 = (int)x[xi++];

  while(1) {
    // Find intersecting indices
    if (xj2 >= yj1 && yj2 >= xj1) {
      int j1 = MAX(xj1,yj1);
      int j2 = MIN(xj2,yj2);
      int xp = xi + j1 - xj1;
      int yp = yi + j1 - yj1;
      for (int k=0;k < j2-j1+1;k++) {
	res += (double)x[xp++] * (double)y[yp++];
      }
    }
    // Increment x or y pointer
    if (yj2 <= xj2) {
      if (++yb >= ynum) break;
      yi += yj2-yj1+1;
      yj1 = y[yi++];
      yj2 = y[yi++];
    } else {
      if (++xb >= xnum) break;
      xi += xj2-xj1+1;
      xj1 = x[xi++];
      xj2 = x[xi++];
    }
  }
  return res;
}

static inline void c_add(double *W, const float* x, const double a) {
  int xp = 1;
  // Iterate through blocks, and grab boundary indices using matlab's indexing
  cout << x[0] << x[1] << x[2] << x[3] << x[4] << "\n";
  cout << "block : " << x[0] << ' ';
  for (int b = 0; b < x[0]; b++) {
    int wp  = (int)x[xp++];
    int len = (int)x[xp++] - wp;
    cout << "wp : " << wp << " len : " << len << ' ';
    for (int i = 0; i < len; i++) {
      cout << x[xp] << ' ';
      W[wp++] += a * (double)x[xp++];
    }
    //printf("(%d,%d)",wp,len);
  }
  cout << '\n';
}

// idC(idP)[i] is the sum of alpha value for examples with ID(:,I(i))
// idI[i] is a pointer to some example with the same id as example I[i]
void sumAlpha(int32_t *ID, const double* A, const int *I,double *idC, int *idP, int *idI, int len) {
  mm = m + sizeof(int)/sizeof(int32_t);
  int32_t *sID = (int32_t *)calloc(mm*n,sizeof(int32_t));
 // Create a matrix of selected ids (given with matlab indexing)
 // that will be sorted, appending the position at end
  for (int j = 0; j < n; j++) {
    int j1 =  I[j]*mm;
    for (int i = 0; i < m; i++) {
      sID[j1+i] = ID[j+i*len];
    }
    INDEX(sID,j) = j;
  }

    for (int j = 0; j < n; j++) {
      for (int i = 0; i < mm; i++) {
        cout << sID[j*mm+i];
      }
      cout << '\n';
    }

  cout << "--------------------- \n";

  // Sort
  qsort(sID,n,mm*sizeof(int32_t),comp);

      for (int j = 0; j < n; j++) {
        for (int i = 0; i < mm; i++) {
          cout << sID[j*mm+i];
        }
        cout << '\n';
      }

  // Go through sorted list, adding alpha values of examples with identical ids
  int i0  = I[INDEX(sID,0)];
  int num = 0;
  for (int t = 0; t < n; t++) {
    int j  = INDEX(sID,t);
    int i1 = I[j];
    if (comp( sID + m*i1, sID + m*i0))
      num++;
    idP[j]    = num;
    idC[num] += A[i1];
    i0       = i1;
    if (A[i1] > 0) {
      idI[num] = i1;
    }
  }
  free(sID);
}
