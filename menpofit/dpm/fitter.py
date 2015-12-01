from menpo.shape import Tree
import numpy as np


def compute_unary_scores(feature_pyramid, filters):
    from menpo.feature.gradient import convolve_python_f  # TODO: define a more proper file for it
    unary_scores = []
    for feat in feature_pyramid: # for each level in the pyramid
        resp = convolve_python_f(feat, filters)
        unary_scores.append(resp)
    return unary_scores


def compute_pairwise_scores(scores, tree, def_coef, anchor):
    from menpo.feature.gradient import call_shiftdt   # TODO: define a more proper file for it.
    Iy, Ix = {}, {}
    for depth in range(tree.maximum_depth - 1, 0, -1):
        for curr_vert in tree.vertices_at_depth(depth):
            (Ny, Nx) = scores[tree.parent(curr_vert)].shape
            w = def_coef[curr_vert]
            cy = anchor[curr_vert][0]
            cx = anchor[curr_vert][1]
            msg, Ix1, Iy1 = call_shiftdt(scores[curr_vert], cx, cy, Nx, Ny, 1)
            scores[tree.parent(curr_vert)] += msg
            Ix[curr_vert] = np.copy(Ix1)
            Iy[curr_vert] = np.copy(Iy1)
    return scores


class DPM_fit():
    def __init__(self, tree, filters, def_coef, anchor):
        assert(isinstance(tree, Tree))
        assert(len(filters) == len(tree.vertices) == len(def_coef))
        self.tree = tree
        self.filters = filters
        self.def_coef = def_coef
        self.anchor = anchor
        
    def fit(self, image):
        from menpo.feature.gradient import convolve_python_f  # TODO: define a more proper file for it.
        feats, scales = featpyramid(image, 5, 4, (3, 3))
        
        for feat in feats:  # for each level in the pyramid
            unary_scores = convolve_python_f(feat, self.filters)
            
            scores = compute_pairwise_scores(np.copy(unary_scores), 
                                             self.tree, self.def_coef, self.anchor)
            

        
def featpyramid(im, m_inter, sbin, pyra_pad):
    # construct the featpyramid. For the time being, it has similar conventions as in 
    # Ramanan's code. TODO: In the future, use menpo's gaussian pyramid.
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


def get_components(model, pyra_pad):
    # implementation of the modelcomponents to return the components
    components = []
    comp_m = model['components'][0][0][0] # *_m = * in matlab
    for c in range(len(comp_m)):
        cm = comp_m[c][0]
        comp_parts = []
        for k in range(len(cm)):
            p = {} 
            p['defid'] = cm[k]['defid'][0][0]
            p['filterid'] = cm[k]['filterid'][0][0]
            p['parent'] = cm[k]['parent'][0][0] - 1
            _shape = model['filters'][0][0][0][p['filterid'] - 1][0].shape  # -1 due to python numbering
            p['sizy'] = _shape[0] 
            p['sizx'] = _shape[1]
            p['filterI'] = model['filters'][0][0][0][p['filterid'] - 1][1][0][0]  # -1 due to python numbering

            x = model['defs'][0][0][0][p['defid'] - 1]  # -1 due to python numbering
            p['w'] = 1 * x['w'][0]  # http://stackoverflow.com/a/6435446
            (ax, ay, ds) = np.copy(x['anchor'][0])
            p['starty'] = ay
            p['startx'] = ax

            comp_parts.append(dict(p))
        components.append(list(comp_parts)) # in outer loop 
    return components


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
    import menpo.io as mio
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
    filters_all = get_filters(model)
    _ms = model['maxsize'][0][0][0]
    sbin = model['sbin'][0][0][0][0]
    pyra_pad = (max(_ms[1] - 2, 0), max(_ms[0] - 2, 0))  # (padx, pady)
    components = get_components(model, pyra_pad)
    
    # random component, chosen for debugging
    parts = components[3]
    def_coef = []  # deformation coefficients
    filters = []  # filters for the component chosen
    anchor = []
    for c, pk in enumerate(parts):
        filters.append(filters_all[pk['filterid'] - 1])
        if c == 0:  # root is on 0 by default in Ramanan
            (w1, w2, w3, w4) = (0., 0., 0., 0.)
        else:
            (w1, w2, w3, w4) = pk['w']
        def_coef.append((w1, w2, w3, w4))
        anchor.append((p['starty'], p['startx']))
        
    im = mio.import_builtin_asset.einstein_jpg()

    