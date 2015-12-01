from menpo.shape import Tree
import numpy as np

class DPM_fit():
    def __init__(self, tree, filters, distance_coef, bias):
        assert(isinstance(tree, Tree))
        self.tree = tree

def featpyramid(im, m_inter, sbin, pyra_pad):
    # construct the featpyramid. For the time being, it has similar conventions as in Ramanan's code. In the
    # future, use menpo's gaussian pyramid.
    from math import log as log_m
    from menpo.feature import hog
    sc = 2 ** (1. / m_inter)
    imsize = (im.shape[0], im.shape[1])
    max_scale = int(log_m(min(imsize)*1./(5*sbin))/log_m(sc)) + 1
    feats, scales = {}, {}
    for i in range(m_inter):
        sc_l = 1. / sc ** i
        scaled = im.rescale(sc_l)
        feats[i] = hog(scaled, mode='sparse', algorithm='zhuramanan', cell_size=sbin/2)
        scales[i] = 2 * sc_l
        feats[i + m_inter] = hog(scaled, mode='sparse', algorithm='zhuramanan', cell_size=sbin)
        scales[i + m_inter] = sc_l

        for j in range(i + m_inter, max_scale, m_inter):
            scaled = scaled.rescale(0.5)
            feats[j + m_inter] = hog(scaled, mode='sparse', algorithm='zhuramanan', cell_size=sbin)
            scales[j + m_inter] = 0.5 * scales[j]

    feats_np = {}  # final feats (keep only numpy array, get rid of the image)
    for k, val in scales.iteritems():
        scales[k] = sbin * 1. / val
        feats_np[k] = np.pad(feats[k].pixels, ((0, 0), (pyra_pad[1] + 1, pyra_pad[1] + 1),
                                               (pyra_pad[0] + 1, pyra_pad[0] + 1)), 'constant')

    return feats_np, scales


def copy_to_new_array(arr):
    # copies each value of an original array to a new one in the same shape.
    # Due to matlab having column-major.
    sh = arr.shape
    new = np.empty(sh, order='C')
    for c in range(sh[0]):
        for i in range(sh[1]):
            for j in range(sh[2]):
                new[c, i, j] = np.copy(arr[c, i, j])
    return new


def get_filters(model):
    ff = model['filters'][0][0][0]
    filters = []
    for i in range(len(ff)):
        tt = np.rollaxis(ff[i]['w'], 2)  # rollaxis, as long as matlab model is loaded (channels in front).
        tt = tt[:-1, :, :]  # hack as long as matlab hog is loaded (32 vs 31 feats in python).
        tt2 = copy_to_new_array(tt)
        filters.append(tt2)
    return filters


def debugging():
    # temp debugging
    # define the tree
    from os.path import isdir
    import scipy.io
    from scipy.sparse import csr_matrix
    m = csr_matrix(([1] * 67, (
                           [0,  0,  0,  0,  1,  3,  5,  6,  7,  8,  8,  9,  9, 10, 12, 13, 14,
                            15, 16, 17, 18, 20, 20, 21, 23, 24, 25, 26, 27, 28, 29, 31, 31, 31,
                            32, 32, 33, 33, 38, 38, 39, 39, 40, 43, 44, 45, 46, 46, 47, 48, 50,
                            51, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66],
                           [1,  3,  5, 31,  2,  4,  6,  7,  8,  9, 20, 10, 12, 11, 13, 14, 15,
                            16, 17, 18, 19, 21, 23, 22, 24, 25, 26, 27, 28, 29, 30, 32, 37, 38,
                            33, 36, 34, 35, 39, 42, 40, 41, 43, 44, 45, 46, 47, 50, 48, 49, 51,
                            52, 60, 53, 54, 55, 56, 57, 58, 59, 61, 62, 63, 64, 65, 66, 67])), shape=(68, 68))
    tree = Tree(m, root_vertex=0, skip_checks=True)   # ## check

    # import the mat file (with the model).
    pm = '/vol/atlas/homes/grigoris/external/dpm_ramanan/face-release1.0-basic/'
    assert(isdir(pm))
    file1 = pm + 'face_p146_small.mat'
    mat = scipy.io.loadmat(file1)

    model = mat['model']
    filters= get_filters(model)