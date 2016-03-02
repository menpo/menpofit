import menpo.io as mio
import numpy as np
import scipy.io
import sys
import time
from menpo.shape import Tree
from menpofit.dpm import DPMLearner, DPMFitter, DPM, non_max_suppression_fast, clip_boxes, bb_to_lns
from scipy.sparse import csr_matrix
from os.path import isdir


def debugging():

    # dpm_learner = DPMLearner()

    pm = '/vol/atlas/homes/grigoris/external/dpm_ramanan/face-release1.0-basic/'
    assert(isdir(pm))
    file1 = pm + 'face_p146_small.mat'
    mat = scipy.io.loadmat(file1)

    # Convert matlab model into python model format
    mat_model = mat['model']
    maxsize = mat_model['maxsize'][0][0][0]
    sbin = mat_model['sbin'][0][0][0][0]
    interval = mat_model['interval'][0][0][0][0]
    thresh = min(0, mat_model['thresh'][0][0][0][0])
    components, anchors = get_components(mat_model)
    filters_all = get_filters(mat_model)
    defs_all = get_defs(mat_model, anchors)
    im = mio.import_builtin_asset.takeo_ppm()

    mat_model = dict()
    mat_model['maxsize'] = maxsize
    mat_model['sbin'] = sbin
    mat_model['interval'] = interval
    mat_model['filters'] = filters_all
    mat_model['defs'] = defs_all
    mat_model['components'] = components
    mat_model = DPM.model_from_dict(mat_model)

    # Uncomment these to use the model learned from python

    # pickle_dev = '/vol/atlas/homes/ks3811/pickles/refactor'
    # try:
    #     import os
    #     fp = os.path.join(pickle_dev, 'actual_parts_model_fast.pkl')
    #     model = mio.import_pickle(fp)
    #     model['interval'] = 10  # Use deeper pyramid when detecting actual objects
    #     mat_model = DPM.model_from_dict(model)
    # except ValueError:  # pickle does not exist
    #     pass

    start = time.time()
    boxes = DPMFitter().fast_fit_from_model(im, mat_model, thresh)
    print 'Found {0} configuration(s) that score above a given threshold: {1}'.format(np.size(boxes), thresh)
    boxes.sort(key=lambda item: item['s'], reverse=True)
    cc, pick = non_max_suppression_fast(clip_boxes(boxes), 0.3)
    lns = bb_to_lns(boxes, pick)
    end = time.time()
    print 'Fitting time taken: {0} seconds'.format(end - start)
    return lns, im


def get_components(model):
    # Component contains informations about filters and defs indexes as well as each component tree structure
    components = []
    comp_m = model['components'][0][0][0] # *_m = * in matlab
    anchors = []
    for c in range(len(comp_m)):    # 13 components
        cm = comp_m[c][0]
        def_ids = []
        filter_ids = []
        parents = []
        # def_index = []
        # filter_index = []
        num_parts = len(cm)
        for k in range(num_parts):
            def_id = cm[k]['defid'][0][0]
            def_ids.append(def_id - 1)
            filter_id = cm[k]['filterid'][0][0]
            filter_ids.append(filter_id - 1)
            parents.append(cm[k]['parent'][0][0] - 1)
            # def_index.append(model['defs'][0][0][0][def_id - 1][1][0][0])  # -1 due to python numbering
            # filter_index.append(model['filters'][0][0][0][filter_id - 1][1][0][0])  # -1 due to python numbering
            x = model['defs'][0][0][0][def_id - 1]  # -1 due to python numbering
            (ax, ay, ds) = np.copy(x['anchor'][0])
            anchors.append((ax, ay, ds))
        pairs = zip(parents, range(num_parts))
        tree_matrix = csr_matrix(([1] * (num_parts-1), (zip(*pairs[1:]))), shape=(num_parts, num_parts))
        tree = Tree(tree_matrix, root_vertex=0, skip_checks=True)
        component = dict()
        component['def_ids'] = def_ids
        component['filter_ids'] = filter_ids
        component['tree'] = tree
        # component['def_index'] = def_index
        # component['filter_index'] = filter_index
        components.append(component)
    return components, anchors


def get_filters(model):
    ff = model['filters'][0][0][0]
    filters = []
    for i in range(len(ff)):
        tt = np.rollaxis(ff[i]['w'], 2)  # rollaxis since matlab format is (5, 5, 32) while python hog is (31, 5, 5)
        tt = tt[:-1, :, :]  # matlab hog is loaded (32 vs 31 feats in python).
        tt2 = copy_to_new_array(tt)
        filters.append({'w': tt2})
    return filters


def copy_to_new_array(arr):
    # Copies each value of an original array to a new one in the same shape due to matlab having column-major.
    sh = arr.shape
    new = np.empty(sh, order='C')
    for c in range(sh[0]):
        for i in range(sh[1]):
            for j in range(sh[2]):
                new[c, i, j] = np.copy(arr[c, i, j])
    return new


def get_defs(model, anchors):
    model_defs_all = model['defs'][0][0][0]
    defs_all = []
    for i in range(len(model_defs_all)):
        defs = model_defs_all[i]['w'][0]
        defs_all.append({'w': defs, 'anchor': anchors[i]})
    return defs_all


if __name__ == "__main__":
    sys.exit(debugging())
