# distutils: language = c++

import numpy as np
cimport numpy as np
cimport cython


cdef extern from "cpp/fconv.h":
    void *process(double *A, double *B, double *C, int r_a, int c_a, int r_b, int c_b, int r_c, int c_c, int num_feat)

cdef extern from "cpp/score.h":
    double cscore(double *A, float *B)

cdef extern from "cpp/shiftdt.h":
    void dt1d(double *src, double *dst, int *ptr, int step,
              int len, double a, double b, int dshift, int dlen, double dstep)

cdef extern from "cpp/qp_one_sparse.h":
    void setMN(int M, int N)

cdef extern from "cpp/qp_one_sparse.h":
    void sumAlpha(int *ID, double *A, int * I, double * idC, int * idP, int * idI, int len)

cdef extern from "cpp/qp_one_sparse.h":
    double c_one_score(double *W, float *X)

cdef extern from "cpp/qp_one_sparse.h":
    double dot(float *W, float *X)

cdef extern from "cpp/qp_one_sparse.h":
    double c_add(double *W, float *X, double a)

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b
cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef convolve_python_f(np.ndarray[double, ndim=3] A, list listB):
    cdef:
        list C = []
        np.ndarray[double, ndim=3] B
        np.ndarray[double, ndim=2] out1
        int f, num_feat = A.shape[0], r_B, c_B, r_C, c_C
        int r_A = A.shape[1], c_A = A.shape[2]
    for B in listB:
        r_B = B.shape[1]
        c_B = B.shape[2]
        r_C = r_A - r_B + 1
        c_C = c_A - c_B + 1
        assert(r_C >= 0 and c_C >= 0)
        out1 = np.zeros((r_C, c_C), order='C')   # should be zeros instead of empty, otherwise c code might be 'random'.
        process(&A[0,0,0], &B[0,0,0], &out1[0,0], r_A, c_A, r_B, c_B, r_C, c_C, num_feat)
        C.append(out1)
    return C

def call_shiftdt(double[:, :] score, double[:] w, int startx, int starty, int Nx, int Ny, int step):
    # note: search for the gradient_cython (np.ndarray thing in input call.)
    # call_shiftdt(child['score'], -child['w'], child['startx'], child['starty'], Nx, Ny, child['step'] )
    # w = -w  (when called in python !!!!!!!!!!!!!)
    cdef int sizy = score.shape[0]
    cdef int sizx = score.shape[1]
    cdef double ax, bx, ay, by
    ax, bx, ay, by = w
    eps = 10 ** (-7)
    # assert( abs(ax) > eps)  # ax, ay should be non-zero.
    # assert( abs(ay) > eps)
    startx -= 1   # due to python numbering (conversion to c).
    starty -= 1   # due to python numbering (conversion to c).

    cdef np.ndarray[double, ndim=2] M = np.zeros((Ny, Nx), dtype=np.double)
    cdef np.ndarray[int, ndim=2] Ix = np.empty((Ny, Nx), dtype=np.int32)
    cdef np.ndarray[int, ndim=2] Iy = np.empty((Ny, Nx), dtype=np.int32)
    cdef np.ndarray[double, ndim=1] tmpM = np.zeros((Nx * sizy, ), dtype=np.double)
    cdef np.ndarray[int, ndim=1] tmpIx = np.zeros((Nx * sizy, ), dtype=np.int32)
    cdef int x, y

    for y in range(sizy):
        dt1d(&score[y, 0], &tmpM[y*Nx], &tmpIx[y*Nx], 1, sizx, ax, bx, startx, Nx, step)

    for x in range(Nx):
        dt1d(&tmpM[x], &M[0,x], &Iy[0, x], Nx, sizy, ay, by, starty, Ny, step)  # the Iy call based on column order of matlab.

    for y in range(Ny):
        for x in range(Nx):
            Ix[y, x] = tmpIx[Iy[y, x]*Nx + x]

    return M, Ix, Iy

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef score(np.ndarray[double, ndim=1] W, np.ndarray[float, ndim=2] X, np.ndarray[int, ndim=1] I):
    cdef:
        int i, l = I.size
        np.ndarray[double, ndim=1] scores = np.zeros((l,), dtype=np.double)
    for i in range(l):
        scores[i] = one_score(X, W, I[i])
    return scores

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef lincomb(np.ndarray[float, ndim=2] X, np.ndarray[double, ndim=1] A, np.ndarray[int, ndim=1] I, int m):
    cdef:
        int i, j, b, xp, wp, len, l = I.size
        float *x
        double a
        np.ndarray[double, ndim=1] w = np.zeros((m,), dtype=np.double)
    for i in range(l):
        a = A[I[i]]
        xp = 1
        for b in range(int(X[0, I[i]])):
            wp = int(X[xp, I[i]])
            xp += 1
            len = int(X[xp, I[i]]) - wp
            xp += 1
            for j in range(len):
                w[wp] += a * X[xp, I[i]]
                wp += 1
                xp += 1
    return w

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef qp_one_sparse(np.ndarray[float, ndim=2] X, np.ndarray[int, ndim=2] ID, np.ndarray[float, ndim=1] B, \
    np.ndarray[double, ndim=1] D, np.ndarray[double, ndim=1] A, np.ndarray[double, ndim=1] W, \
    np.ndarray[int, ndim=2] Noneg, np.ndarray[dtype=np.int8_t, ndim=1] SV, np.ndarray[double, ndim=1] L, double C, \
    np.ndarray[int, ndim=1] I):
    cdef:
        int cnt, i, j, i2
        int k = X.shape[0]
        int p = int_max(Noneg.shape[0], Noneg.shape[1])
        int n = int_max(I.shape[0], I.shape[1])
        int m = int_min(ID.shape[0], ID.shape[1])
        int len = int_max(ID.shape[0], ID.shape[1])
        double Ci, G, PG, dA, maxA, sum
        np.ndarray[double, ndim=1] err = np.zeros((n,), dtype=np.double)
        np.ndarray[double, ndim=1] idC = np.zeros((n,), dtype=np.double)
        np.ndarray[int, ndim=1] idP = np.zeros((n,), dtype=np.intc)
        np.ndarray[int, ndim=1] idI = np.zeros((n,), dtype=np.intc) - 1
    setMN(m, n)
    #print m, n
    #print I
    #print 'start sumAlpha'
    #print 'ID : ', ID[0:93, :]
    sum_alpha(ID, A, I, idC, idP, idI)
    #print 'ID : ', ID[0:93, :]
    #print 'A : ', A
    print 'idC : ', idC[idC > C]
    print 'idP : ', np.shape(idP), idP
    print 'idI : ', np.shape(idP), np.sort(idI)[::-1]
    #print 'done sumAlpha'
    assert(np.all(idC <= C+10**-5))
    assert(np.all(idC >= -10**-5))
    for cnt in range(n):
        i = I[cnt]
        #print 'i : ', i
        j = idP[cnt]
        A[i] = double_max(double_min(A[i], C), 0)
        Ci = double_max(double_min(idC[j], C), A[i])
        assert(Ci <= C+10**-5)
        G = one_score(X, W, i) - B[i]
        if i < 100:
            print 'G : ', G
        PG = G
        if (A[i] == 0 and G >= 0) or (Ci >= C and G <= 0):
            PG = 0
        if -G > err[j]:
            err[j] = -G
        if A[i] == 0 and G > 0:
            SV[i] = False
        if Ci >= C and G < -10**-12 and A[i] < C and idI[j] != i and idI[j] >= 0:
            print 'con1'
            i2 = idI[j]
            G -= one_score(X, W, i2) - B[i2]
            if A[i] == 0 and G >0:
                G = 0
                SV[i] = False
            if G > 10**-12 or G < -10**-12:
                #dA = -G / (D[i] + D[i2] - 2 * dot(&X[0, i], &X[0, i2]))
                dA = -G / (D[i] + D[i2] - 2 * np.dot(np.transpose(np.asmatrix(X[0, i])), X[0, i2]))
                dA2 = -G / (D[i] + D[i2] - 2 * np.sum(X[0, i] * X[0, i2]))
                assert(np.all(dA == dA2))
                if dA > 0:
                    dA = double_min(double_min(dA, C - A[i]), A[i2])
                else :
                    dA = double_max(double_max(dA, -A[i]), A[i2] - C)
                A[i] += dA
                A[i2] -= dA
                L[0] += dA * (B[i] - B[i2])
                #print 'L : ', L[0]
                assert(A[i]  >= 0 and A[i]  <= C);
                assert(A[i2] >= 0 and A[i2] <= C);
                add(X, W, dA, i)
                add(X, W, -dA, i2)
                #print 'W : ', W
                for d in range(p):
                    W[Noneg[d]] = double_max(W[Noneg[d]], 0)
        elif PG > 10**-12 or PG < -10**-12:
            #print 'con2'
            dA = A[i]
            assert(dA <= Ci+10**-5)
            maxA = double_max(C - Ci + dA, 0)
            A[i] = double_min(double_max(A[i] - G/D[i], 0), maxA)
            assert(A[i] >= 0 and A[i] <= C)
            dA = A[i] - dA
            #print 'dA : ', dA
            L[0] += dA * B[i]
            #print 'L : ', L[0]
            #print 'X : ', X[i, 0:10]
            idC[j] = double_min(double_max(Ci + dA, 0), C)
            #print "x size : ", X.shape[0], X.shape[1]
            add(X, W, dA, i)
            #print 'W : ', W
            for d in range(p):
                W[Noneg[d]] = double_max(W[Noneg[d]], 0)
            assert(idC[j] >= 0 and idC[j] <= C)
        if A[i] > 0:
            #print 'con3'
            idI[j] = i
    sum = 0
    for i in range(n):
        sum += err[i]
    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef add(np.ndarray[float, ndim=2] X, np.ndarray[double, ndim=1] W, double A, int i):
    cdef:
        int xp, b, j
    xp = 1
    #print "block : ", X[0, i]
    for b in range(int(X[0, i])):
        wp = int(X[xp, i])
        #print "wp : ", wp
        xp += 1
        len = int(X[xp, i]) - wp
        #print "len : ", len
        xp += 1
        for j in range(len):
            W[wp] += A * X[xp, i]
            #print 'x : ', X[xp, i]
            wp += 1
            xp += 1

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef one_score(np.ndarray[float, ndim=2] X, np.ndarray[double, ndim=1] W, int i):
    cdef:
        int xp, b, j
        double y
    xp = 1
    y = 0
    for b in range(int(X[0, i])):
        wp = int(X[xp, i])
        xp += 1
        len = int(X[xp, i]) - wp
        xp += 1
        for j in range(len):
            y += W[wp] * X[xp, i]
            wp += 1
            xp += 1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef sum_alpha(np.ndarray[int, ndim=2] ID, np.ndarray[double, ndim=1] A, np.ndarray[int, ndim=1] I, \
    np.ndarray[double, ndim=1] idC, np.ndarray[int, ndim=1] idP, np.ndarray[int, ndim=1] idI):
    cdef:
        int num = 0, i0, i1
        np.ndarray[int, ndim=2] ID2
        np.ndarray[long, ndim=1] J
    ID2 = ID[:, I]
    J = np.lexsort((ID2[0, :], ID2[1, :], ID2[2, :], ID2[3, :], ID2[4, :]))
    i0 = I[J[0]]
    for j in J:
        i1 = I[j]
        if np.any(ID[:, i0] != ID[:, i1]):
            num += 1
        idP[j] = num
        idC[num] += A[i1]
        i0 = i1
        if A[i1] > 0:
            idI[num] = i1
