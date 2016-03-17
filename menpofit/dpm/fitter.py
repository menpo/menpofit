from __future__ import print_function
import numpy as np
import gc
from .utils import convolve_python_f, call_shiftdt
from math import log as log_m
from menpo.feature import hog


class DPMFitter(object):

    r"""
    def _fit(self, feats, scales, padding, threshold=-1, filters=None, defs=None, anchors=None, bias=None, tree=None):
        from menpo.feature.gradient import convolve_python_f  # TODO: define a more proper file for it.

        # define filter size in [y, x] format. Assumption: All filters
        # have the same size, otherwise pass a list in backtrack.
        fsz = [filters[0].shape[1], filters[0].shape[2]]
        boxes = []  # list with detection boxes (as dictionaries)
        for level, feat in feats.iteritems():  # for each level in the pyramid
            unary_scores = convolve_python_f(feat, filters)

            scores, Ix, Iy = _compute_pairwise_scores(np.copy(unary_scores), tree, defs, anchors)

            scale = scales[level]
            rscore = scores[tree.root_vertex] + bias

            [Y, X] = np.where(rscore > threshold)

            if X.shape[0] > 0:
                XY = _backtrack(X, Y, tree, Ix, Iy, fsz, scale, padding)

            for i in range(X.shape[0]):
                x, y = X[i], Y[i]
                detection_info = dict()
                detection_info['level'] = level
                detection_info['s'] = np.copy(rscore[y, x])
                detection_info['xy'] = XY[:, :, i]

                boxes.append(dict(detection_info))

        return boxes

    def fast_fit(self, image, threshold=-1):
        from menpo.feature.gradient import convolve_python_f  # TODO: define a more proper file for it.
        padding = (3, 3)  # TODO: param in the future maybe?
        feats, scales = _featpyramid(image, 5, 4, padding)

        boxes = []  # list with detection boxes (as dictionaries)
        filters_all = self._model.filters_all
        defs_all = self._model.defs_all
        components = self._model.components
        fsz = [filters_all[0].shape[1], filters_all[0].shape[2]]

        for level, feat in feats.iteritems():  # for each level in the pyramid
            unary_scores_all = convolve_python_f(feat, filters_all)
            for c, component in enumerate(components):
                tree = component['tree']
                anchors = component['anchors']
                bias = component['bias']
                filter_ids = component['filter_ids']
                def_ids = component['def_ids']
                scores, Ix, Iy = _fast_compute_pairwise_scores(np.copy(unary_scores_all), defs_all, tree, filter_ids,
                                                               def_ids, anchors, padding)
                scale = scales[level]
                root_id = filter_ids[tree.root_vertex]
                rscore = scores[root_id] + bias
                [Y, X] = np.where(rscore > threshold)

                if X.shape[0] > 0:
                    XY = old_backtrack(X, Y, tree, Ix, Iy, fsz, scale, padding)  # todo: comback seprate backtrack from writing to ex

                for i in range(X.shape[0]):
                    x, y = X[i], Y[i]
                    box = XY[:, :, i]
                    detection_info = dict()
                    detection_info['level'] = level
                    detection_info['s'] = np.copy(rscore[y, x])
                    detection_info['xy'] = box
                    boxes.append(dict(detection_info))
        return boxes
    """

    @staticmethod
    def fast_fit_from_model(image, model, threshold=-10**100):
        # fit the DPM model to the given image
        padding = (model.maxsize[0]-1-1, model.maxsize[1]-1-1)
        feats, scales = _feature_pyramid(image, model.interval, model.sbin, padding)

        boxes = []
        filters_all = model.get_filters_weights()
        defs_all = model.get_defs_weights()
        anchors_all = model.get_defs_anchors()

        for level, feat in feats.iteritems():  # for each level in the feature pyramid
            unary_scores_all = convolve_python_f(feat, filters_all)
            for c, component in enumerate(model.components):

                if not component:
                    continue

                tree = component['tree']
                filter_ids = component['filter_ids']
                def_ids = component['def_ids']
                defs = np.array(defs_all)[def_ids]
                bias = defs[tree.root_vertex]
                anchors = np.array(anchors_all)[def_ids]
                fsz = np.array(filters_all[filter_ids[tree.root_vertex]].shape)[1:]

                scores = np.array(unary_scores_all)[filter_ids]
                scores, ix, iy = _fast_compute_pairwise_scores(scores, defs, tree, anchors, padding)
                scale = scales[level]
                root_score = scores[tree.root_vertex] + bias
                [ys, xs] = np.where(root_score > threshold)

                if xs.shape[0] > 0:
                    xy = _old_backtrack(xs, ys, tree, ix, iy, fsz, scale, padding)

                for i in range(xs.shape[0]):
                    x, y = xs[i], ys[i]
                    detection_info = dict()
                    detection_info['level'] = level
                    detection_info['s'] = np.copy(root_score[y, x])
                    detection_info['xy'] = xy[:, :, i]
                    boxes.append(dict(detection_info))

        return boxes

    @staticmethod
    def fit_with_bb(image, model, threshold=-10**100, bbox=dict(), overlap=0, id_=0, label=0, qp=None):
        # fit the DPM model to the given image
        latent = 'box' in bbox and len(bbox['box']) > 0
        ex = dict()  # ex is the example that will e written to qp. maybe ex can be moved to qp instead
        ex['blocks'] = []
        ex['id'] = np.array([label, id_, 0, 0, 0])
        padding = (model.maxsize[0]-1-1, model.maxsize[1]-1-1)
        feats, scales = _feature_pyramid(image, model.interval, model.sbin, padding)

        boxes = []  # list with detection boxes (as dictionaries)
        filters_all = model.get_filters_weights()
        filters_index_all = model.get_filters_indexes()
        defs_all = model.get_defs_weights()
        defs_index_all = model.get_defs_indexes()
        anchors_all = model.get_defs_anchors()

        for level, feat in feats.iteritems():  # for each level in the pyramid
            unary_scores_all = convolve_python_f(feat, filters_all)
            for c, component in enumerate(model.components):

                if not component:
                    continue
                if latent:
                    if c != bbox['c']:
                        continue

                tree = component['tree']
                filter_ids = component['filter_ids']
                def_ids = component['def_ids']
                defs = np.array(defs_all)[def_ids]
                bias = defs[tree.root_vertex]
                anchors = np.array(anchors_all)[def_ids]
                filter_index = np.array(filters_index_all)[filter_ids]
                def_index = np.array(defs_index_all)[def_ids]
                fsz = np.array(filters_all[filter_ids[tree.root_vertex]].shape)[1:]

                # for positive examples, only allow the configurations that mostly overlap with actual annotations
                ovmask = {}
                if latent:
                    skip_flag = False
                    for cv in tree.vertices:
                        # assume each part have a same filter size
                        ovmask[cv] = _test_overlap(fsz, feat, scales[level], padding, image.shape, bbox['box'][cv, :],
                                                   overlap)
                        if not np.any(ovmask[cv]):
                            skip_flag = True
                            break
                    if skip_flag:
                        continue

                scores = np.array(unary_scores_all)[filter_ids]
                for i, ov in ovmask.iteritems():
                    assert(np.any(ov))
                    mask = np.zeros(ov.shape)
                    mask[np.logical_not(ov)] = -999999  # can not be -np.inf because of using distance transform
                    scores[i] = scores[i] + mask

                scores, ixs, iys = _fast_compute_pairwise_scores(scores, defs, tree, anchors, padding)
                scale = scales[level]
                root_score = scores[tree.root_vertex] + bias

                if latent:  # only pick the best configuration for positive example
                    threshold = max(threshold, np.max(root_score))

                [ys, xs] = np.where(root_score >= threshold)

                # todo: test if it faster to use old_backtrack
                import time
                start = time.time()
                if xs.shape[0] > 0:
                    # Only backtrack examples that can be fitted into qp for this negative image
                    ex_allowed_num = min(np.size(xs), qp.nmax - qp.n)
                    rand_choices = np.random.choice(np.size(xs), ex_allowed_num)
                    xs = xs[rand_choices]
                    ys = ys[rand_choices]
                    xys, ids, ex_filters, ex_defs = _batch_backtrack(xs, ys, tree, ixs, iys, fsz, scale, padding, level,
                                                                     feat, anchors, label, id_)

                print('level :', level, 'component :', c, 'found :', xs.shape[0])
                if not latent and xs.shape[0] > 0:
                    qp.write_multiple_exs(ids, filter_index, def_index, ex_filters, ex_defs)
                for i in range(xs.shape[0]):
                    x, y = xs[i], ys[i]
                    box = xys[:, :, i]
                    detection_info = dict()
                    detection_info['level'] = level
                    detection_info['s'] = np.copy(root_score[y, x])
                    detection_info['xy'] = box  # xy
                    boxes.append(dict(detection_info))
                    if not latent:
                        # qp.write(ex)
                        qp.increment_neg_ub(root_score[y, x])
                if xs.shape[0] > 0 and not latent:
                    del ids, ex_filters, ex_defs
                    gc.collect()
                stop = time.time()
                print('time taken :', stop-start)

                if not latent and xs.shape[0] > 0:  # and qp.n < qp.nmax:
                    assert(np.allclose(qp.score_neg(), root_score[y, x]))

                if not latent and (qp.lb < 0 or 1-qp.lb/qp.ub > 0.05 or qp.n == qp.nmax):
                    model = qp.obtimise(model)
                    filters_all = model.get_filters_weights()
                    unary_scores_all = convolve_python_f(feat, filters_all)
                    defs_all = model.get_defs_weights()
                    anchors_all = model.get_defs_anchors()

        if latent:
            qp.write_multiple_exs(ids, filter_index, def_index, ex_filters, ex_defs)
            if np.size(boxes) > 0:
                boxes = boxes[-1]
                assert(np.allclose(qp.score_pos()[qp.n-1], boxes['s']))
        return boxes, model

    @staticmethod
    def fit_with_bb_less_memory(image, model, threshold=-10**100, bbox=dict(), overlap=0, id_=0, label=0, qp=None):
        # fit the DPM model to the given image, consume less memory but it is significantly slower than above function.
        latent = 'box' in bbox and len(bbox['box']) > 0
        ex = dict()  # ex is the example that will e written to qp. maybe ex can be moved to qp instead
        ex['blocks'] = []
        ex['id'] = np.array([label, id_, 0, 0, 0])
        padding = (model.maxsize[0]-1-1, model.maxsize[1]-1-1)
        feats, scales = _feature_pyramid(image, model.interval, model.sbin, padding)

        boxes = []  # list with detection boxes (as dictionaries)
        filters_all = model.get_filters_weights()
        filters_index_all = model.get_filters_indexes()
        defs_all = model.get_defs_weights()
        defs_index_all = model.get_defs_indexes()
        anchors_all = model.get_defs_anchors()

        for level, feat in feats.iteritems():  # for each level in the pyramid
            unary_scores_all = convolve_python_f(feat, filters_all)
            for c, component in enumerate(model.components):

                if not component:
                    continue
                if latent:
                    if c != bbox['c']:
                        continue

                tree = component['tree']
                filter_ids = component['filter_ids']
                def_ids = component['def_ids']
                defs = np.array(defs_all)[def_ids]
                bias = defs[tree.root_vertex]
                anchors = np.array(anchors_all)[def_ids]
                filter_index = np.array(filters_index_all)[filter_ids]
                def_index = np.array(defs_index_all)[def_ids]
                fsz = np.array(filters_all[filter_ids[tree.root_vertex]].shape)[1:]

                # for positive examples, only allow the configurations that mostly overlap with actual annotations
                ovmask = {}
                if latent:
                    skip_flag = False
                    for cv in tree.vertices:
                        # assume each part have a same filter size
                        ovmask[cv] = _test_overlap(fsz, feat, scales[level], padding, image.shape,
                                                          bbox['box'][cv, :], overlap)
                        if not np.any(ovmask[cv]):
                            skip_flag = True
                            break
                    if skip_flag:
                        continue

                scores = np.array(unary_scores_all)[filter_ids]
                for i, ov in ovmask.iteritems():
                    assert(np.any(ov))
                    mask = np.zeros(ov.shape)
                    mask[np.logical_not(ov)] = -999999  # can not be -np.inf because of using distance transform
                    scores[i] = scores[i] + mask

                scores, ixs, iys = _fast_compute_pairwise_scores(scores, defs, tree, anchors, padding)
                scale = scales[level]
                root_score = scores[tree.root_vertex] + bias

                if latent:  # only pick the best configuration for positive example
                    threshold = max(threshold, np.max(root_score))

                [ys, xs] = np.where(root_score >= threshold)

                # todo: test if it faster to use old_backtrack
                import time
                start = time.time()

                print('level :', level, 'component :', c, 'found :', xs.shape[0])
                for i in range(xs.shape[0]):
                    x, y = xs[i], ys[i]
                    xy = _backtrack(x, y, tree, ixs, iys, fsz, scale, padding, ex, True, level, filter_index, def_index,
                                    feat, anchors)
                    detection_info = dict()
                    detection_info['level'] = level
                    detection_info['s'] = np.copy(root_score[y, x])
                    detection_info['xy'] = xy
                    boxes.append(dict(detection_info))
                    if not latent:
                        qp.write(ex)
                stop = time.time()
                print('time taken :', stop-start)

                if not latent and xs.shape[0] > 0 and qp.n < qp.nmax:
                    assert(np.allclose(qp.score_neg(), root_score[y, x]))

                if not latent and (qp.lb < 0 or 1-qp.lb/qp.ub > 0.05 or qp.n == qp.nmax):
                    model = qp.obtimise(model)
                    filters_all = model.get_filters_weights()
                    unary_scores_all = convolve_python_f(feat, filters_all)
                    defs_all = model.get_defs_weights()
                    anchors_all = model.get_defs_anchors()

        if latent:
            qp.write(ex)
            if np.size(boxes) > 0:
                boxes = boxes[-1]
                assert(np.allclose(qp.score_pos()[qp.n-1], boxes['s']))
        return boxes, model


def _test_overlap(fsz, feat, scale, padding, img_size, box, overlap):
    # return the 2d boolean matrix with the same size of each part score indicating if each position overlap with the
    # bounding box more than a given threshold
    [bx1, by1, bx2, by2] = box
    (_, dim_y, dim_x) = feat.shape
    (size_x, size_y) = fsz
    (padx, pady) = padding
    (imgx, imgy) = img_size
    x1 = (np.array([range(0, dim_x - size_x + 1)]) - padx - 1) * scale + 1
    y1 = (np.array([range(0, dim_y - size_y + 1)]) - pady - 1) * scale + 1
    x2 = x1 + size_x*scale - 1
    y2 = y1 + size_y*scale - 1
    x1 = np.minimum(np.maximum(x1, np.zeros_like(x1)), np.full_like(x1, imgx-1))
    x2 = np.maximum(np.minimum(x2, np.full_like(x2, imgx-1)), np.zeros_like(x2))
    y1 = np.minimum(np.maximum(y1, np.zeros_like(y1)), np.full_like(y1, imgy-1))
    y2 = np.maximum(np.minimum(y2, np.full_like(y2, imgy-1)), np.zeros_like(y2))
    xx1 = np.maximum(x1, np.full_like(x1, bx1))
    xx2 = np.minimum(x2, np.full_like(x2, bx2))
    yy1 = np.maximum(y1, np.full_like(y1, by1))
    yy2 = np.minimum(y2, np.full_like(y2, by2))
    w = xx2 - xx1 + 1
    h = yy2 - yy1 + 1
    w[w < 0] = 0
    h[h < 0] = 0
    inter = np.dot(h.T, w)
    area = np.dot((y2-y1+1).T, x2-x1+1)
    box = np.dot((by2-by1+1).T, bx2-bx1+1)
    ov = inter / (area + box - inter)
    return np.asarray(ov > overlap)


def _fast_compute_pairwise_scores(scores, def_coefs, tree, anchors=None, padding=None):
    r"""
    Given the (unary) scores it computes the pairwise scores by utilising the Generalised Distance
    Transform.

    Parameters
    ----------
    scores: `list`
        The (unary) scores to which the pairwise score will be added.
    tree: `:map:`Tree``
        Tree with the parent/child connections.
    def_coef: `list`
        Each element contains a 4-tuple with the deformation coefficients for that part.
    anchors:
        Contains the anchor position in relation to the parent of each part.

    Returns
    -------
    scores: `ndarray`
        The (unary + pairwise) scores.
    Ix: `dict`
        Contains the coordinates of x for each part from the Generalised Distance Transform.
    Iy: `dict`
        Contains the coordinates of y for each part from the Generalised Distance Transform.
    """
    iy, ix = {}, {}
    for depth in range(tree.maximum_depth, 0, -1):
        for curr_vert in tree.vertices_at_depth(depth):
            parent = tree.parent(curr_vert)
            (ny, nx) = scores[parent].shape
            w = def_coefs[curr_vert] * -1.00
            (cx, cy, ds) = anchors[curr_vert]
            step = 2**ds
            virt_padx = (step - 1)*padding[0]
            virt_pady = (step - 1)*padding[1]
            startx = cx - virt_padx
            starty = cy - virt_pady
            msg, ix1, iy1 = call_shiftdt(scores[curr_vert], np.array(w, dtype=np.double), startx, starty, nx, ny, 1)
            scores[parent] += msg
            ix[curr_vert] = np.copy(ix1)
            iy[curr_vert] = np.copy(iy1)
    return scores, ix, iy


def compute_filters_scores_only(scores, tree, filter_ids, ptr):
    # might be needed when debugging
    new_scores = []
    for depth in range(0, tree.maximum_depth + 1):
        for curr_vert in tree.vertices_at_depth(depth):
            x, y = ptr[curr_vert]
            curr_filter_id = filter_ids[curr_vert]
            new_scores.append(scores[curr_filter_id][y, x])
    return new_scores


def _feature_pyramid(im, m_inter, sbin, pyra_pad):
    # Construct the feature pyramid. For the time being, it has similar conventions as in Ramanan's code.
    # TODO: In the future, use menpo's gaussian pyramid.
    sc = 2 ** (1. / m_inter)
    img_size = (im.shape[0], im.shape[1])
    max_scale = int(log_m(min(img_size)*1./(5*sbin))/log_m(sc)) + 1
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


def _old_backtrack(x, y, tree, ix, iy, fsz, scale, pyra_pad):
    r"""
    Backtrack the solution from the root to the children and return the detection coordinates for each part.
    algorithm:
    (for all points that are greater than the threshold -> vectorised)
    start from the root
    save location of the ln point
    for depth in range(1, max_depth_graph): # traverse tree from the root to the leaves
        curr_ch_vertext = [vertices in current depth]
        for each vertex in curr_ch_vertext:
            find from Ix, Iy the position of the child
            save ideal location of the ln point
    Parameters
    ----------
    x: `list`
        The x coordinates of the detections in the root level.
    y: `list`
        The y coordinates of the detections in the root level.
    tree: `:map:`Tree``
        Tree with the parent/child connections.
    ix: `dict`
        Contains the coordinates of x for each part from the Generalised Distance Transform.
    iy: `dict`
        Contains the coordinates of y for each part from the Generalised Distance Transform.
    fsz: `list`
        The size of the filter in the format of [y, x].
        ASSUMPTION: All filters at this level have the same size. Otherwise,
        modify the code for a list of filter sizes.
    scale: `int`
        The scale of the detections in the feature pyramid.
    pyra_pad: `list`
        Padding in the form of [x, y] in the feature pyramid.
    Returns
    -------
    box: `ndarray`
        The x, y coordinates of the detection(s).
    """
    numparts = len(tree.vertices)
    ptr = np.empty((numparts, 2, x.shape[0]), dtype=np.int64)
    box = np.empty((numparts, 4, x.shape[0]))
    k = tree.root_vertex

    ptr[k, 0, :] = np.copy(x)
    ptr[k, 1, :] = np.copy(y)
    box[k, 0, :] = (ptr[k, 0, :] - pyra_pad[0]) * scale + 1
    box[k, 1, :] = (ptr[k, 1, :] - pyra_pad[1]) * scale + 1
    box[k, 2, :] = box[k, 0, :] + fsz[1] * scale - 1
    box[k, 3, :] = box[k, 1, :] + fsz[0] * scale - 1

    for depth in range(1, tree.maximum_depth + 1):
        for cv in tree.vertices_at_depth(depth):  # for each vertex in that level
            par = tree.parent(cv)
            x = ptr[par, 0, :]
            y = ptr[par, 1, :]
            # idx = np.ravel_multi_index((y, x), dims=Ix[cv].shape, order='C')
            # ptr[cv, 0, :] = Ix[cv].ravel()[idx]
            # ptr[cv, 1, :] = Iy[cv].ravel()[idx]
            ptr[cv, 0, :] = ix[cv][y, x]
            ptr[cv, 1, :] = iy[cv][y, x]

            box[cv, 0, :] = (ptr[cv, 0, :] - pyra_pad[0]) * scale + 1
            box[cv, 1, :] = (ptr[cv, 1, :] - pyra_pad[1]) * scale + 1
            box[cv, 2, :] = box[cv, 0, :] + fsz[1] * scale - 1
            box[cv, 3, :] = box[cv, 1, :] + fsz[0] * scale - 1
    return box


def _batch_backtrack(xs, ys, tree, ixs, iys, fsz, scale, pyra_pad, level=-1, feat=None, anchors=None, label=-1, ex_id=-1):
    r"""
    Backtrack the solution from the root to the children and return the detection coordinates for each part.
    The backtrack is done for each rot position in parallel making the computationfaster but consume more memory.

    algorithm:
    (for all points that are greater than the threshold -> vectorised)
    start from the root
    save location of the ln point
    for depth in range(1, max_depth_graph): # traverse tree from the root to the leaves
        curr_ch_vertext = [vertices in current depth]
        for each vertex in curr_ch_vertext:
            find from Ix, Iy the position of the child
            save ideal location of the ln point

    Parameters
    ----------
    xs: `list`
        The x coordinates of the detections in the root level.
    ys: `list`
        The y coordinates of the detections in the root level.
    tree: `:map:`Tree``
        Tree with the parent/child connections.
    ixs: `dict`
        Contains the coordinates of x for each part from the Generalised Distance Transform.
    iys: `dict`
        Contains the coordinates of y for each part from the Generalised Distance Transform.
    fsz: `list`
        The size of the filter in the format of [y, x].
        ASSUMPTION: All filters at this level have the same size. Otherwise,
        modify the code for a list of filter sizes.
    scale: `int`
        The scale of the detections in the feature pyramid.
    pyra_pad: `list`
        Padding in the form of [x, y] in the feature pyramid.
    level: `int`
        Level of the example in the feature pyramid.
    feat: `ndarray`
        Features of the corresponding level in feature pyramid.
    anchors: `list`
        A list of anchors of each part(ideal locations of each part reference from its parent).
    label: `int`
        A boolean indicating if the examples are positive or negative.
    ex_id: `int`
        An id of the current example

    Returns
    -------
    box: `ndarray`
        The x, y coordinates of the detection(s).
    ids: `ndarray`
        The corresponding id of each example
    filters: `ndarray`
        The filters of each example filters.shape = [num_parts, feature.size (31*5*5), ex_num]
    defs: `ndarray`
        The defs of each example defs.shape = [num_parts, defs.size (4), ex_num]
    """
    xs = np.array(xs)
    ys = np.array(ys)
    ex_num = xs.shape[0]
    num_parts = len(tree.vertices)
    ptr = np.empty((num_parts, 2, ex_num), dtype=np.int64)
    box = np.empty((num_parts, 4, ex_num))
    cv = tree.root_vertex

    filters = np.zeros((num_parts, np.prod(fsz)*feat.shape[0], ex_num,), dtype=np.float32)
    defs = np.zeros((num_parts, 4, ex_num,), dtype=np.float32)  # 4 = dx^2, dx, dy^2, dy

    ids = np.zeros((5, np.size(xs)), dtype=np.float32)
    ids[0, :] = label
    ids[1, :] = ex_id
    ids[2, :] = level
    ids[3] = xs + round(fsz[0]/2)
    ids[4] = ys + round(fsz[1]/2)

    ptr[cv, 0, :] = np.copy(xs)
    ptr[cv, 1, :] = np.copy(ys)
    box[cv, 0, :] = (ptr[cv, 0, :] - pyra_pad[0]) * scale + 1
    box[cv, 1, :] = (ptr[cv, 1, :] - pyra_pad[1]) * scale + 1
    box[cv, 2, :] = box[cv, 0, :] + fsz[1] * scale - 1
    box[cv, 3, :] = box[cv, 1, :] + fsz[0] * scale - 1

    for i in range(np.size(xs)):
        filters[cv, :, i] = feat[:, ys[i]:ys[i]+fsz[0], xs[i]:xs[i]+fsz[1]].ravel()

    for depth in range(1, tree.maximum_depth + 1):
        for cv in tree.vertices_at_depth(depth):  # for each vertex in that level
            par = tree.parent(cv)
            xs = ptr[par, 0, :]
            ys = ptr[par, 1, :]
            ptr[cv, 0, :] = ixs[cv][ys, xs]
            ptr[cv, 1, :] = iys[cv][ys, xs]

            box[cv, 0, :] = (ptr[cv, 0, :] - pyra_pad[0]) * scale + 1
            box[cv, 1, :] = (ptr[cv, 1, :] - pyra_pad[1]) * scale + 1
            box[cv, 2, :] = box[cv, 0, :] + fsz[1] * scale - 1
            box[cv, 3, :] = box[cv, 1, :] + fsz[0] * scale - 1

            defs[cv, :, :] = _def_vector(xs, ys, ptr[cv, 0, :], ptr[cv, 1, :], pyra_pad, anchors[cv])
            xs = ptr[cv, 0, :]
            ys = ptr[cv, 1, :]
            for i in range(np.size(xs)):
                filters[cv, :, i] = feat[:, ys[i]:ys[i]+fsz[0], xs[i]:xs[i]+fsz[1]].ravel()
    return box, ids, filters, defs


def _backtrack(x, y, tree, ix, iy, fsz, scale, pyra_pad, ex=None, write=False, level=-1, filter_index=None,
               def_index=None, feat=None, anchors=None):
    r"""
    Backtrack the solution from the root to the children and return the detection coordinates for each part.

    algorithm:
    (for all points that are greater than the threshold -> vectorised)
    start from the root
    save location of the ln point
    for depth in range(1, max_depth_graph): # traverse tree from the root to the leaves
        curr_ch_vertext = [vertices in current depth]
        for each vertex in curr_ch_vertext:
            find from Ix, Iy the position of the child
            save ideal location of the ln point

    Parameters
    ----------
    x: `list`
        The x coordinates of the detections in the root level.
    y: `list`
        The y coordinates of the detections in the root level.
    tree: `:map:`Tree``
        Tree with the parent/child connections.
    ix: `dict`
        Contains the coordinates of x for each part from the Generalised Distance Transform.
    iy: `dict`
        Contains the coordinates of y for each part from the Generalised Distance Transform.
    fsz: `list`
        The size of the filter in the format of [y, x].
        ASSUMPTION: All filters at this level have the same size. Otherwise,
        modify the code for a list of filter sizes.
    scale: `int`
        The scale of the detections in the feature pyramid.
    pyra_pad: `list`
        Padding in the form of [x, y] in the feature pyramid.
    level: `int`
        Level of the example in the feature pyramid.
    filter_index: `list`
        List of corresponding filters index reference to qp.w
    def_index: `list`
        List of corresponding defs index reference to qp.w
    feat: `ndarray`
        Features of the corresponding level in feature pyramid.
    anchors: `list`
        A list of anchors of each part(ideal locations of each part reference from its parent).

    Returns
    -------
    box: `ndarray`
        The x, y coordinates of the detection(s).
    """
    numparts = len(tree.vertices)
    ptr = np.empty((numparts, 2), dtype=np.int64)
    box = np.empty((numparts, 4))
    k = tree.root_vertex

    ptr[k, 0] = np.copy(x)
    ptr[k, 1] = np.copy(y)
    box[k, 0] = (ptr[k, 0] - pyra_pad[0]) * scale + 1
    box[k, 1] = (ptr[k, 1] - pyra_pad[1]) * scale + 1
    box[k, 2] = box[k, 0] + fsz[0] * scale - 1
    box[k, 3] = box[k, 1] + fsz[1] * scale - 1
    if write:
        ex['id'][2:] = [level, round(x + fsz[0]/2), round(y + fsz[1]/2)]
        ex['blocks'] = []
        block = dict()
        block['i'] = def_index[k]
        block['x'] = 1  # to match with bias i.e. 1 * bias
        ex['blocks'].append(block)
        block = dict()
        block['i'] = filter_index[k]
        block['x'] = feat[:, y:y+fsz[0], x:x+fsz[1]]
        ex['blocks'].append(block)
    for depth in range(1, tree.maximum_depth + 1):
        for cv in tree.vertices_at_depth(depth):  # for each vertex in that level
            par = tree.parent(cv)
            x = ptr[par, 0]
            y = ptr[par, 1]
            idx = np.ravel_multi_index((y, x), dims=ix[cv].shape, order='C')
            ptr[cv, 0] = ix[cv].ravel()[idx]
            ptr[cv, 1] = iy[cv].ravel()[idx]

            box[cv, 0] = (ptr[cv, 0] - pyra_pad[0]) * scale + 1
            box[cv, 1] = (ptr[cv, 1] - pyra_pad[1]) * scale + 1
            box[cv, 2] = box[cv, 0] + fsz[0] * scale - 1
            box[cv, 3] = box[cv, 1] + fsz[1] * scale - 1
            if write:
                block = dict()
                block['i'] = def_index[cv]
                block['x'] = _def_vector(x, y, ptr[cv, 0], ptr[cv, 1], pyra_pad, anchors[cv])
                ex['blocks'].append(block)
                block = dict()
                block['i'] = filter_index[cv]
                x = ptr[cv, 0]
                y = ptr[cv, 1]
                block['x'] = feat[:, y:y+fsz[0], x:x+fsz[1]]
                ex['blocks'].append(block)
    return box


def _def_vector(px, py, x, y, padding, anchor):
    # calculate distance between a part position and the the part's parent anchor position
    (ax, ay, ds) = anchor
    step = 2**ds
    virt_padx = (step - 1)*padding[0]
    virt_pady = (step - 1)*padding[1]
    startx = ax - virt_padx
    starty = ay - virt_pady
    probex = (px - 1)*step + startx
    probey = (py - 1)*step + starty
    dx = probex - x
    dy = probey - y
    return -dx**2, -dx, -dy**2, -dy


def non_max_suppression_fast(boxes, overlap_thresh):
    # Malisiewicz et al. method.
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return [], []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    sc = boxes[:, 4]  # score confidence

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(sc)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_thresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int"), pick


def clip_boxes(boxes):
    # convert the boxes in a format suitable for non_maximum_suppression
    bb = np.empty((len(boxes), 5))
    for i, bb_s in enumerate(boxes):
        bb[i, 0] = np.min(bb_s['xy'][:, 0])
        bb[i, 1] = np.min(bb_s['xy'][:, 1])
        bb[i, 2] = np.max(bb_s['xy'][:, 2])
        bb[i, 3] = np.max(bb_s['xy'][:, 3])
        bb[i, 4] = np.max(bb_s['s'])
    return bb


def bb_to_lns(boxes, pick):
    # convert the selected boxes into lns (the centre of each bbox).
    lns1 = []
    for cnt, p in enumerate(pick):
        b = boxes[p]['xy']
        pts = np.empty((b.shape[0], 2))
        pts[:, 1] = np.int32((b[:, 0] + b[:, 2])/2)  # !!! reverse x, y due to menpo convention (first y axis!)
        pts[:, 0] = np.int32((b[:, 1] + b[:, 3])/2)
        lns1.append(pts)
    return lns1
