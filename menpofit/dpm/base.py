import time
import os
import numpy as np
from numpy import size, nonzero
from scipy.sparse import csr_matrix
import scipy.io

import menpo.io as mio
from menpo.feature import hog
from menpo.image import Image
from menpo.shape import Tree

from .fitter import DPMFitter  # , non_max_suppression_fast, clip_boxes, bb_to_lns
from .utils import score, lincomb, qp_one_sparse


class DPMLearner(object):
    def __init__(self, config=None):
        # assert(isinstance(tree, Tree))
        # assert(len(filters) == len(tree.vertices) == len(def_coef))
        # self.components = components
        # self.filters_all = filters_all
        # self.defs_all = defs_all
        self.config = config

        # remove when config is passed properly or can be learned from the example.
        if self.config is None:
            self.config = self._get_face_default_config()

        self.fitter = DPMFitter()
        start = time.time()
        self._model_train()
        stop = time.time()
        print stop-start
        print 'done'

    @staticmethod
    def _get_face_default_config():
        conf = dict()  # configurations for the mixture components.
        conf['sbin'] = 4
        conf['viewpoint'] = range(90, -90-15, -15)  # 90 <-> -90
        conf['partpoolsize'] = 39 + 68 + 39
        conf['mixture_poolid'] = [range(0, 39), range(0, 39), range(0, 39),
                                  range(39, 107), range(39, 107), range(39, 107), range(39, 107),
                                  range(39, 107), range(39, 107), range(39, 107),
                                  range(107, 146), range(107, 146), range(107, 146)]
        return conf

    @staticmethod
    def _get_parents_lns(gmixid):
        # Parents of each landmark point as defined in the original code.  # todo: declare tree and get rid of this.
        if 0 <= gmixid <= 2:
            parents = np.array(([0, 1, 2, 3, 4, 5,
                                 1, 7, 8, 9, 10,
                                 11, 12, 13, 14,
                                 1, 16, 17, 18, 19, 20, 21,
                                 19, 23, 24, 23, 26,
                                 22, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]))
        elif 3 <= gmixid <= 9:
            parents = np.array(([0, 1, 2, 1, 4,
                                 1, 6, 7, 8,
                                 9, 10, 11, 10, 13, 14,
                                 15, 16, 17, 18, 19,
                                 9, 21, 22, 21, 24, 25,
                                 26, 27, 28, 29, 30,
                                 1, 32, 33, 34, 34, 33, 32, 32, 39, 40, 40, 39,
                                 41, 44, 45, 46, 47, 48, 49,
                                 47,
                                 51, 52, 53, 54, 55, 56, 57, 58, 59, 52, 61, 62, 63, 64, 65, 66, 67]))
        elif 10 <= gmixid <= 12:
            parents = np.array(([0, 1, 2, 3, 4, 5,
                                 1, 7, 8, 9, 10,
                                 11, 12, 13, 14,
                                 1, 16, 17, 18, 19, 20, 21,
                                 19, 23, 24, 23, 26,
                                 22, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]))
        else:
            raise ValueError('No such model for parents\' list exists.')
        return parents - 1  # -1 to change matlab indexes

    @staticmethod
    def _get_anno2tree(gmixid):
        if 0 <= gmixid <= 2:
            anno2tree = np.array(([6, 5, 4, 3, 2, 1,
                                   14, 15, 11, 12, 13,
                                   10, 9, 8, 7,
                                   16, 17, 18, 19, 20, 21, 22,
                                   25, 24, 23, 26, 27,
                                   28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]))
        elif 3 <= gmixid <= 9:
            anno2tree = np.array(([34, 33, 32, 35, 36,
                                   31, 30, 29, 28,
                                   40, 41, 42, 39, 38, 37,
                                   18, 19, 20, 21, 22,
                                   43, 48, 47, 44, 45, 46,
                                   27, 26, 25, 24, 23,
                                   52, 51, 50, 49, 61, 62, 63, 53, 54, 55, 65, 64,
                                   56, 66, 57, 67, 59, 68, 60, 58,
                                   9,8 , 7, 6, 5, 4, 3, 2, 1, 10, 11, 12, 13, 14, 15, 16, 17]))
        elif 10 <= gmixid <= 12:
            anno2tree = np.array(([6, 5, 4, 3, 2, 1,
                                   14, 15, 11, 12, 13,
                                   10, 9, 8, 7,
                                   16, 17, 18, 19, 20, 21, 22,
                                   25, 24, 23, 26, 27,
                                   28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39]))
        else:
            raise ValueError('No such model for anno2tree\' list exists.')
        return anno2tree - 1  # -1 to change matlab indexes

    @staticmethod
    def _get_pie_image_info(pos_data, pos_data_dir, anno_dir, neg_data_dir):
        pos = []
        train_list = [50, 50, 50, 50, 50, 50, 300, 50, 50, 50, 50, 50, 50]
        print('Collecting info for the positive images.')
        for gmixid, poses in enumerate(pos_data):
            count = 0
            for img in poses['images']:
                if count >= train_list[gmixid]:
                    break
                img_name = img[0][0]
                [sub_id, ses_id, rec_id, cam_id, _] = img_name.split('_')
                cam1_id = cam_id[0:2]
                cam2_id = cam_id[2]
                file_name = '{0}/session{1}/png/{2}/{3}/{4}_{5}/{6}.png'.format(pos_data_dir, ses_id, sub_id, rec_id, cam1_id, cam2_id, img_name)
                anno_file_name = '{0}/{1}_lm.mat'.format(anno_dir, img_name)
                if not os.path.isfile(file_name) or not os.path.isfile(anno_file_name):
                    continue
                count += 1
                aux = dict()
                aux['pts'] = scipy.io.loadmat(anno_file_name)['pts'][DPMLearner._get_anno2tree(gmixid), :]
                aux['im'] = file_name
                aux['gmixid'] = gmixid
                pos.append(aux)
        print('Collecting info for the negative images.')
        l1 = sorted(os.listdir(neg_data_dir))
        neg = []
        for elem in l1:
            if elem[elem.rfind('.') + 1:] in ['jpg', 'png', 'jpeg']:
                aux = dict()
                aux['im'] = os.path.join(neg_data_dir, elem)
                neg.append(dict(aux))
        return pos, neg

    def _model_train(self):
        multipie_mat = '/homes/ks3811/Phd/1 year/wildFace/face-release1.0-full/multipie.mat'
        multipie = scipy.io.loadmat(multipie_mat)['multipie'][0]

        multipie_dir = '/vol/hci2/Databases/video/MultiPIE'
        anno_dir = '/homes/ks3811/Phd/1 year/wildFace/face-release1.0-full/my_annotation'
        # negims = '/vol/hmi/projects/christos/Journal_IVC_2013/03_Ramanan/INRIA/'
        neg_imgs_dir = '/homes/ks3811/Phd/1 year/wildFace/face-release1.0-full/INRIA'
        pickle_dev = '/vol/atlas/homes/ks3811/pickles/refactor'

        try:  # Check if the data is already existed.
            fp = os.path.join(pickle_dev, 'data.pkl')
            _c = mio.import_pickle(fp)
            pos = _c['pos']
            neg = _c['neg']
        except ValueError:  # Data in pickle does not exist
            start = time.time()
            pos, neg = self._get_pie_image_info(multipie, multipie_dir, anno_dir, neg_imgs_dir)
            _c = dict()
            _c['pos'] = pos
            _c['neg'] = neg
            mio.export_pickle(_c, fp)
            stop = time.time()
            fp = os.path.join(pickle_dev, 'data_time.pkl')
            mio.export_pickle(stop-start, fp)

        pos = self._ln2box(pos)
        spos = self._split(pos)
        k = min(len(neg), 200)
        kneg = neg[0:k]

        #todo : Learning each tndependent part can be done in parallel
        #pool = mp.Pool(processes=4)
        #results = pool.map(self._train_model, range(1,7))
        #results = [pool.apply(self._train_model, args=(x,)) for x in range(1, 7)]
        #print results

        parts_models = []
        try:
            fp = os.path.join(pickle_dev, 'actual_parts_model_fast.pkl')
            parts_models = mio.import_pickle(fp)
        except ValueError:
            start = time.time()
            for i in xrange(self.config['partpoolsize']):
                assert(len(spos[i]) > 0)
                init_model = self._init_model(spos[i], self.config['sbin'])
                parts_models.append(self._train(init_model, spos[i], kneg, iters=4))
            fp = os.path.join(pickle_dev, 'actual_parts_model_fast.pkl')
            mio.export_pickle(parts_models, fp)
            stop = time.time()
            fp = os.path.join(pickle_dev, 'actual_parts_model_fast_time.pkl')
            mio.export_pickle(stop-start, fp)

        try:  # todo: if independent parts are learned in parallel, need to wait for all results before continuing.
            fp = os.path.join(pickle_dev, 'defs.pkl')
            defs = mio.import_pickle(fp)
        except ValueError:
            defs = self.build_mixture_defs(pos, parts_models[0].maxsize[0])
            fp = os.path.join(pickle_dev, 'defs.pkl')
            mio.export_pickle(defs, fp)

        try:
            fp = os.path.join(pickle_dev, 'mix.pkl')
            model = mio.import_pickle(fp)
        except ValueError:
            model = self.build_mixture_model(parts_models, defs)
            start = time.time()
            model = self._train(model, pos, kneg, 1)
            fp = os.path.join(pickle_dev, 'mix.pkl')
            mio.export_pickle(model, fp)
            stop = time.time()
            fp = os.path.join(pickle_dev, 'mix_time.pkl')
            mio.export_pickle(stop-start, fp)

        try:
            fp = os.path.join(pickle_dev, 'final.pkl')
            model = mio.import_pickle(fp)
        except ValueError:
            start = time.time()
            model = self._train(model, pos, neg, 2)
            fp = os.path.join(pickle_dev, 'final.pkl')
            mio.export_pickle(model, fp)
            stop = time.time()
            fp = os.path.join(pickle_dev, 'final.pkl')
            mio.export_pickle(stop-start, fp)

    def _split(self, pos_images):
        # Each component contains different number of parts.
        # Split the positive exampls's parts according to its part pools.
        conf = self.config
        split_pos = []
        for i in range(conf['partpoolsize']):
            split_pos.append([])
        for p in pos_images:
            part_ids_inpool = conf['mixture_poolid'][p['gmixid']]
            for i, k in enumerate(part_ids_inpool):
                s = dict()
                s['im'] = p['im'][:]
                s['gmixid'] = 0  # Each independent part is assume to belong to 0th component.
                s['box_y1'] = [p['box_y1'][i]]
                s['box_x1'] = [p['box_x1'][i]]
                s['box_y2'] = [p['box_y2'][i]]
                s['box_x2'] = [p['box_x2'][i]]
                split_pos[k].append(dict(s))
        return split_pos

    def _ln2box(self, pos):
        # Converts the points in bboxes. (same as point2box.mat in the original code)
        for i, p in enumerate(pos):
            parents_lns = self._get_parents_lns(p['gmixid'])
            lengths = np.linalg.norm(abs(p['pts'][1:] - p['pts'][parents_lns[1:]]), axis=1)
            boxlen = np.percentile(lengths, 80, interpolation='midpoint')  # 0.73% to compensate for the zero norm 34 point.
            assert(boxlen > 3)  # ensure that boxes are 'big' enough.
            _t = np.clip(p['pts'] - 1 - boxlen/2, 0, np.inf) # -1 for matlab indexes
            p['box_x1'] = np.copy(_t[:, 0])
            p['box_y1'] = np.copy(_t[:, 1])
            _t = p['pts'] - 1 + boxlen/2  # no check for boundary, -1 for matlab indexes
            p['box_x2'] = np.copy(_t[:, 0])
            p['box_y2'] = np.copy(_t[:, 1])
        return pos

    def _init_model(self, pos_, sbin):
        areas = np.empty((len(pos_),))
        for i, el in enumerate(pos_):
            areas[i] = (el['box_x2'][0] - el['box_x1'][0] + 1) * (el['box_y2'][0] - el['box_y1'][0] + 1)
        areas = np.sort(areas)  # Pick 20 percentile area
        area = areas[np.floor(areas.size * 0.2)]  # todo: return to 0.2. Current value of 0.21 just to make sz 5 and easier to work with
        nw = np.sqrt(area)

        im = hog(Image(np.zeros((1, 30, 30))), mode='sparse', algorithm='zhuramanan')  # Calculating HOG features size
        siz = [im.pixels.shape[0], round(nw/sbin), round(nw/sbin)]

        d = dict()  # deformation
        d['w'] = 0  # bias
        d['i'] = 0
        d['anchor'] = np.zeros((3,))

        f = dict()  # filter
        f['w'] = np.empty(siz)
        f['i'] = 1

        c = dict()
        c['filter_ids'] = [0]
        c['def_ids'] = [0]
        c['parent'] = -1
        c['anchors'] = [np.zeros((3,))]
        c['tree'] = Tree(np.array([0]), root_vertex=0, skip_checks=True)  # Independent part is a single node tree.

        model = dict()
        model['defs'] = [d]
        model['filters'] = [f]
        model['components'] = [c]

        model['maxsize'] = siz[1:]
        model['len'] = 1 + np.prod(siz)
        model['interval'] = 10
        model['sbin'] = sbin

        model = self._poswarp(model, pos_)
        return DPM.model_from_dict(model)
        # c = model['components'][0]
        # c['bias'] = d['w']  # todo: dont think this is needed

    def _poswarp(self, model, pos):
        # Update independent part model's filter by averaging its hog feature.
        warped = self._warppos(model, pos)
        s = model['filters'][0]['w'].shape  # filter size
        num = len(warped)
        feats = np.empty((np.prod(s), num))

        for c, im in enumerate(warped):
            feat = hog(im, mode='sparse', algorithm='zhuramanan', cell_size=model['sbin'])
            feats[:, c] = feat.pixels.flatten()

        w = np.mean(feats, 1)
        scores = np.sum(w * w)
        w2 = w.reshape(s)
        model['filters'][0]['w'] = np.copy(w2)
        model['obj'] = -scores
        return model

    def _warppos(self, model, pos):
        # Load the images, crop and resize them to a predefined shape.
        f = model['components'][0]['filter_ids'][0]  # potentially redundant, f == 0 (but check the rest first.)
        siz = model['filters'][f]['w'].shape[1:]
        sbin = model['sbin']
        pixels = [sbin * siz[0], sbin * siz[1]]
        num_pos = len(pos)
        heights = np.empty((num_pos,))
        widths = np.empty((num_pos,))
        for i, el in enumerate(pos):
            heights[i] = el['box_y2'][0] - el['box_y1'][0] + 1
            widths[i] = el['box_x2'][0] - el['box_x1'][0] + 1
        crop_size = ((siz[0] + 2) * sbin, (siz[1] + 2) * sbin)
        warped = []
        for i, p in enumerate(pos):
            padx = sbin * widths[i] / pixels[1]
            pady = sbin * heights[i] / pixels[0]

            im = mio.import_image(p['im'])
            im_cr = im.crop([p['box_y1'][0] - pady, p['box_x1'][0] - padx], [p['box_y2'][0] + pady, p['box_x2'][0] + padx])
            im2 = im_cr.resize(crop_size)
            warped.append(im2.copy())
        return warped

    def _train(self, model, pos, negs, iters, c=0.002, wpos=2, wneg=1, maxsize=4, overlap=0.6):
        # the size of vectorized model
        length = model.sparselen()
        # Maximum number of model that fitted into the cache
        nmax = round(maxsize*0.25*10**9/length)
        qp = Qp(model, length, nmax, c, wpos)
        for t in range(iters):
            # pos = [pos[0]]
            model.delta = self._poslatent(t, model, qp, pos, overlap)
            if model.delta < 0.001:
                break
            qp.svfix = range(qp.n)
            qp.sv[qp.svfix] = True
            qp.prune()
            qp.opt(0.5)
            model = qp.vec2model(model)
            model.interval = 4

            for i, neg in enumerate(negs):
                print 'iter:', t, 'neg:', i, '/', np.size(negs)
                im = mio.import_image(neg['im'])
                box, model = self.fitter.fit_with_bb(im,  model, -1, [], 0, i, -1, qp)
                if np.sum(qp.sv) == nmax:
                    break

            qp.opt()
            model = qp.vec2model(model)
            model.lb = qp.lb
            model.ub = qp.ub

        return model

    def _poslatent(self, t, model, qp, poses, overlap):
        num_pos = np.size(poses)
        model.interval = 5
        num_positives = np.zeros(np.size(model.components, ), dtype=int)
        score0 = qp.score_pos()
        qp.n = 0
        old_w = qp.w
        qp = qp.model2qp(model)
        assert(scipy.linalg.norm(old_w - qp.w) < 10**-5)
        # w = (model.model2vec()[0] - qp.w0) * qp.wreg
        # assert(scipy.linalg.norm(w - qp.w) < 10**-5)

        for i, pos in enumerate(poses):
            print 'iter:', t, 'pos:', i, '/', num_pos
            num_parts = np.size(pos['box_x1'])
            bbox = dict()
            bbox['box'] = np.zeros((num_parts, 4))
            bbox['c'] = pos['gmixid'] # to link the index from matlab properly
            for p in range(num_parts):
                bbox['box'][p, :] = [pos['box_x1'][p], pos['box_y1'][p], pos['box_x2'][p], pos['box_y2'][p]]        #todo : bbox values are weird fixit
            im = mio.import_image(pos['im'])
            im, bbox['box'] = self._croppos(im, bbox['box'])
            box, model = self.fitter.fit_with_bb(im,  model, -np.inf, bbox, overlap, i, 1, qp)
            if np.size(box) > 0:
                # im.view()     #  for visualization
                # cc, pick = non_max_suppression_fast(clip_boxes([box]), 0.3)
                # lns = bb_to_lns([box], pick)
                # from menpo.shape import PointCloud
                # im.landmarks['TEST'] = PointCloud(lns[0])
                # im2 = im.crop_to_landmarks_proportion(0.2, group='TEST')
                # im2.view_landmarks(render_numbering=True, group='TEST')
                # assert(False)
                num_positives[bbox['c']] += 1
        delta = np.inf
        if t > 0:
            score1 = qp.score_pos()
            loss0 = np.sum(np.maximum(1 - score0, np.zeros_like(score0)))
            loss1 = np.sum(np.maximum(1 - score1, np.zeros_like(score1)))
            assert(loss1 <= loss0)
            if loss0 != 0:
                delta = abs((loss0 - loss1) / loss0)
        assert(qp.n <= qp.x.shape[1])
        assert(np.sum(num_positives) <= 2 * num_pos)
        return delta

    @staticmethod
    def _croppos(im, box):
        x1 = np.min(box[:, 0])
        y1 = np.min(box[:, 1])
        x2 = np.max(box[:, 2])
        y2 = np.max(box[:, 3])
        pad = 0.5 * ((x2 - x1 + 1) + (y2 - y1 + 1))
        x1 = max(0, np.round(x1 - pad))
        y1 = max(0, np.round(y1 - pad))
        x2 = min(im.shape[1] - 1, np.round(x2 + pad))
        y2 = min(im.shape[0] - 1, np.round(y2 + pad))
        cropped_im = im.crop([y1, x1], [y2 + 1, x2 + 1])
        box[:, 0] -= x1
        box[:, 1] -= y1
        box[:, 2] -= x1
        box[:, 3] -= y1
        return cropped_im, box

    def build_mixture_defs(self, pos, maxsize):

        conf = self.config
        pos_len = len(pos)
        gmixids = np.empty((pos_len,), dtype=np.int32)
        boxsize = np.empty((pos_len,))
        for i, p in enumerate(pos):
            boxsize[i] = p['box_x2'][0] - p['box_x1'][0] + 1
            gmixids[i] = p['gmixid'] * 1

        defs = []
        for i in range(len(conf['mixture_poolid'])):
            nparts = len(conf['mixture_poolid'][i])
            par = self._get_parents_lns(i)
            idx = np.where(i == gmixids)[0]
            if np.size(idx) == 0:
                defs.append([])
                continue
            # assert(idx.shape[0] > 10)  # should have 'some' images in each mixid.
            points = np.empty((nparts, 2, idx.shape[0]))

            for j, id_j in enumerate(idx):
                scale0 = boxsize[id_j]/maxsize
                points[:, :, j] = pos[id_j]['pts']/scale0

            deftemp = (points[:, :, :] - points[par, :, :])
            deftemp = np.mean(deftemp, axis=2)
            defs.append(deftemp[1:, :])
        return defs

    def build_mixture_model(self, models, defs):
        # Diverging from the original code in the sense that the root filter IS NOT in position 0, but
        # in it's position within the ln points. Additionally, there is a new field called root_id, which
        # shows the precise position of that root filter.
        conf = self.config
        model_ = {}

        assert(len(defs) == len(conf['mixture_poolid']))
        model_['maxsize'] = models[0].maxsize
        model_['len'] = 0
        model_['interval'] = models[0].interval
        model_['sbin'] = models[0].sbin

        # add the filters
        model_['filters'] = []
        for m in models:    # for each possible parts
            if np.size(m) > 0:
                f = dict()
                f['w'] = m.filters[0]['w']
                f['i'] = model_['len']
                model_['len'] += np.size(f['w'])
                model_['filters'].append(f)

        model_['defs'] = []
        model_['components'] = []
        for i, def1 in enumerate(defs):  # for each component
            component = dict()
            def_ids = []
            if np.size(def1) == 0:
                model_['components'].append(component)
                continue
            nd = np.size(model_['defs'])
            d = dict()
            d['i'] = model_['len']
            d['w'] = np.array([0])
            d['anchor'] = np.zeros(3,)
            model_['defs'].append(d)
            model_['len'] += np.size(d['w'])
            def_ids.append(nd)
            for j, def_j in enumerate(def1):
                nd = np.size(model_['defs'])
                d = dict()
                d['i'] = model_['len']
                d['w'] = np.array([0.01, 0, 0.01, 0])
                d['anchor'] = np.array([round(def_j[0]) + 1, round(def_j[1]) + 1, 0])
                model_['defs'].append(d)
                model_['len'] += np.size(d['w'])
                def_ids.append(nd)
            component['def_ids'] = def_ids
            component['filter_ids'] = conf['mixture_poolid'][i]
            parents = self._get_parents_lns(i)
            num_parts = np.size(parents)
            pairs = zip(parents, range(num_parts))
            tree_matrix = csr_matrix(([1] * (num_parts-1), (zip(*pairs[1:]))), shape=(num_parts, num_parts))
            component['tree'] = Tree(tree_matrix, root_vertex=0, skip_checks=True)
            model_['components'].append(component)

        return DPM.model_from_dict(model_)


class DPM(object):

    def __init__(self, filters=None, defs=None, components=None, interval=10, sbin=5, maxsize=None, len=-1, lb=0,
                 ub=0, delta=0):
        self.filters = filters
        self.defs = defs
        self.components = components
        self.interval = interval
        self.sbin = sbin
        self.maxsize = maxsize
        self.len = len
        self.lb = lb
        self.ub = ub
        self.delta = delta

    @classmethod
    def model_from_dict(cls, model):
        filters = model['filters'] if 'filters' in model else None
        defs = model['defs'] if 'defs' in model else None
        components = model['components'] if 'components' in model else None
        interval = model['interval'] if 'interval' in model else 10
        sbin = model['sbin'] if 'sbin' in model else 5
        maxsize = model['maxsize'] if 'maxsize' in model else None
        len = model['len'] if 'len' in model else -1
        lb = model['lb'] if 'lb' in model else 0
        ub = model['ub'] if 'ub' in model else 0
        return cls(filters, defs, components, interval, sbin, maxsize, len, lb, ub)

    def sparselen(self):
        # Number of entries needed to encode a block-scarce representation
        len1 = -1
        for tempcnt, comp in enumerate(self.components):
            if not comp:
                continue
            numblocks = 0
            feats = np.zeros((self.len,))
            for cv in comp['tree'].vertices:
                x = self.filters[comp['filter_ids'][cv]]
                i1 = x['i']
                i2 = i1 + np.size(x['w']) - 1
                feats[i1:i2+1] = 1
                numblocks += 1

                x = self.defs[comp['def_ids'][cv]]
                i1 = x['i']
                i2 = i1 + np.size(x['w']) - 1
                feats[i1:i2+1] = 1
                numblocks += 1

            n = 1 + 2 * numblocks + int(np.sum(feats))  # 1 maybe used to encode the length itself
            len1 = max(len1, n)
        return len1

    def get_filters_weights(self):
        return self._extract_from_model('filters', 'w')

    def get_filters_indexes(self):
        return self._extract_from_model('filters', 'i')

    def get_defs_weights(self):
        return self._extract_from_model('defs', 'w')

    def get_defs_indexes(self):
        return self._extract_from_model('defs', 'i')

    def get_defs_anchors(self):
        return self._extract_from_model('defs', 'anchor')

    def _extract_from_model(self, field, sub_field):
        sub_fields = []
        for f in getattr(self, field):
            if f:
                sub_fields.append(f[sub_field])
        return sub_fields


class Qp(object):
    # def __init__(self, length, nmax, num_id, c, wpos):
    #     self.x = np.zeros((length, nmax), dtype=np.float32)
    #     self.i = np.zeros((num_id, nmax), dtype=np.int)
    #     self.b = np.zeros((nmax,), dtype=np.float32)
    #     self.d = np.zeros((nmax,))
    #     self.a = np.zeros((nmax,))
    #     self.sv = np.zeros((nmax,), dtype=np.bool)
    #     self.w = np.zeros((length,))
    #     self.l = np.array([0], dtype=np.double)
    #     self.n = 0
    #     self.ub = 0.0
    #     self.lb = 0.0
    #     self.svfix = []
    #     self.cpos = c * wpos
    #     self.cneg = c
    #     self.wreg = wreg
    #     self.w0 = w0
    #     self.non_neg = non_neg
    #     self.w = (w - self.w0) * self.wreg

    def __init__(self, model, length, nmax, c, wpos):
        num_id = 5  # 5 = [label, id, level, posX, poY]
        self.x = np.zeros((length, nmax), dtype=np.float32)
        self.i = np.zeros((num_id, nmax), dtype=np.int)
        self.b = np.zeros((nmax,), dtype=np.float32)
        self.d = np.zeros((nmax,))
        self.a = np.zeros((nmax,))
        self.sv = np.zeros((nmax,), dtype=np.bool)
        self.w = np.zeros((length,))
        self.l = np.array([0], dtype=np.double)
        self.n = 0
        self.ub = 0.0
        self.lb = 0.0
        self.lb_old = 0.0
        self.svfix = []
        self.cpos = c * wpos
        self.cneg = c
        self.wreg = []
        self.w0 = []
        self.non_neg = []
        self.model2qp(model)

    def score_pos(self):
        y = self.i[0, 0:self.n]
        i = np.array(nonzero(y)[0], dtype=np.intc)
        w = self.w + self.w0 * self.wreg
        scores = score(w, self.x, i)/self.cpos
        return scores

    def score_neg(self):
        w = - (self.w + self.w0 * self.wreg)
        scores = score(w, self.x, np.array([self.n - 1],  dtype=np.intc)) / self.cneg  # -1 bc current n point to the next free space
        return scores

    @staticmethod
    def slow_score(w, x):
        xp = 1
        y = 0
        for b in range(x[0]):
            wp = int(x[xp])
            xp += 1
            length = int(x[xp]) - wp
            xp += 1
            for j in range(length):
                y += w[wp] * x[xp]
                wp += 1
                xp += 1
        return y

    def write(self, ex):
        if self.n == np.size(self.a):
            return

        label = ex['id'][0] > 0
        c = self.cpos if label else self.cneg
        #  ids = np.sort([block['i'] for block in ex['blocks']])

        bias = 1
        norm = 0
        i = self.n
        j = 0
        self.x[:, i] = 0    # needed so that x is sparse i.e. zero for filters and defs of other irrelevant components's parts.
        self.x[j, i] = len(ex['blocks'])

        for block in ex['blocks']:
            n = np.size(block['x'])
            i1 = block['i']
            i2 = i1 + n
            ids = range(i1, i2)
            x = np.array(block['x'])
            x = x if label else -x
            x = x.flatten()  # x.reshape(n, 1)
            bias = bias - np.sum(self.w0[ids] * x)
            x = c*x
            x = x/self.wreg[ids]
            self.x[j+1, i] = i1
            self.x[j+2, i] = i2
            self.x[j+3:j+3+i2-i1, i] = x
            norm += np.sum(x * x)  # np.dot(np.transpose(x), x)
            j += 3 + i2 - i1 - 1

        self.d[i] = norm
        self.b[i] = c*bias
        self.i[:, i] = ex['id']
        self.sv[i] = True
        self.n += 1

    def prune(self):
        if np.all(self.sv):
            self.sv = self.a > 0
            self.sv[self.svfix] = True

        idxs = nonzero(self.sv)[0]
        n = np.size(idxs)
        assert(n > 0)

        self.l[0] = 0
        self.w = np.zeros_like(self.w)
        for j in range(n):
            i = idxs[j]
            self.x[:, j] = self.x[:, i]
            self.i[:, j] = self.i[:, i]
            self.b[j] = self.b[i]
            self.d[j] = self.d[i]
            self.a[j] = self.a[i]
            self.sv[j] = self.sv[i]
            self.l[0] += self.b[j]*self.a[j]
            self.w += self._sparse2dense(self.x[:, j])*self.a[j]

        self.sv[range(n)] = True
        self.sv[n:] = False
        self.a[n:] = 0
        self.project_non_neg()
        self.lb = self.update_lb()
        self.n = n

    def update_lb(self):
        return self.l[0] - 0.5 * np.sum(self.w * self.w)

    def update_ub(self, loss):
        return loss + 0.5 * np.sum(self.w * self.w)

    def increment_neg_ub(self, rscore):
        self.ub += self.cneg*max(1+rscore, 0)

    def project_non_neg(self):
        if np.size(self.non_neg) > 0:
            self.w[self.non_neg] = np.maximum(self.w[self.non_neg], np.zeros_like(self.w[self.non_neg]))

    def _sparse2dense(self, x):
        y = np.zeros((np.size(self.w),))
        j = 0
        for i in range(x[0]):
            i1 = x[j+1]
            i2 = x[j+2]
            y[i1:i2] = x[j+3:j+3+i2-i1]
            j += 3 + i2 - i1 - 1
        # if x[0] > 0:  not true in the mixture of components case
        #     assert(i2 == np.size(self.w))
        return y

    def opt(self, tol=0.05, iters=1000):
        self.refresh()
        I = np.array(range(self.n), dtype=np.intc)
        ids = self.i[:, I]
        J = np.lexsort((ids[4, :], ids[3, :], ids[2, :], ids[1, :], ids[0, :]))  # todo:find a better way to sort column
        ids = ids[:, J]
        eqid = np.zeros_like(J, dtype=bool)
        eqid[1:] = np.all(ids[:, 1:] == ids[:, 0:-1], axis=0)
        slack = self.b[I] - score(self.w, self.x, I)
        loss = self.computeloss(slack[J], eqid)
        ub = self.update_ub(loss)
        lb = self.lb
        self.sv[I] = True
        for t in range(iters):
            self.one()
            lb = self.lb
            ub_est = min(self.ub, ub)
            if lb > 0 and 1 - lb/ub_est < tol:
                slack = self.b[I] - score(self.w, self.x, I)
                loss = self.computeloss(slack[J], eqid)
                ub = min(ub, self.update_ub(loss))
                if 1 - lb/ub < tol:
                    break
                self.sv[I] = True
        self.ub = ub

    def refresh(self):
        idxs = nonzero(self.a > 0)[0]
        idxs = idxs[np.argsort(self.a[idxs])] if np.size(idxs) > 0 else np.array([0])
        idxs = np.array(idxs, dtype=np.intc)
        self.l[0] = np.sum(self.b[idxs] * self.a[idxs])
        self.w = lincomb(self.x, self.a, idxs, np.size(self.w))
        #self.slow_lin_comb(idxs)
        self.project_non_neg()
        self.lb_old = self.lb
        self.lb = self.update_lb()
        if self.lb_old != 0:
            assert(self.lb > self.lb_old - 10**-5)

    def slow_lin_comb(self, ids):
        for i in range(np.size(ids)):
            a = self.a[ids[i]]
        xp = 1
        for b in range(self.x[xp, ids[i]]):
            wp = self.x[xp, ids[i]]
            xp += 1
            len = self.x[xp, ids[i]] - wp
            xp += 1
            for j in range(len):
                self.w[wp] += a * self.x[xp, ids[i]]
                wp += 1
                xp += 1

    @staticmethod
    def computeloss(slack, eqid):
        err = np.zeros_like(eqid, dtype=bool)
        for j in range(np.size(eqid)):
            if not eqid[j]:
                i = j
                v = slack[i]
                if v > 0:
                    err[i] = True
                else:
                    v = 0
            elif slack[j] > v:
                err[i] = False
                i = j
                v = slack[i]
                err[i] = True
        return np.sum(slack[err])

    def one(self):
        idxs = nonzero(self.sv)[0]
        np.random.shuffle(idxs)
        idxs = np.array(idxs, dtype=np.intc)
        n = np.size(idxs)
        assert(n > 0)
        #loss = qp_one_sparse(self.x, np.array(self.i, dtype=np.intc), self.b, self.d, self.a, self.w, np.array(self.noneg, dtype=np.intc), self.sv.view(dtype=np.int8), self.l, 1, idxs)
        loss = self.qp_one_sparse(1, idxs)
        self.refresh()
        self.sv[self.svfix] = True
        self.lb_old = self.lb
        self.lb = self.update_lb()
        self.ub = self.update_ub(loss)
        print 'lb : ', self.lb, 'ub : ', self.ub
        assert(np.all(self.w[self.non_neg] >= 0))
        assert(np.all(self.a[range(self.n)] >= -10**-5))
        assert(np.all(self.a[range(self.n)] <= 1 + 10**-5))

    def mult(self, tol=0.001, iters=1000):
        self.refresh()
        lb = -np.inf
        for t in range(iters):
            init = np.all(self.sv[range(self.n)])
            self.one()
            if lb > 0 and (self.lb - lb)/self.lb < tol:
                if init:
                    break
                self.sv[range(self.n)] = True
            lb = self.lb

    def actual_w(self):
        return self.w / self.wreg + self.w0

    def qp_one_sparse(self, c, I):
        n = np.size(I)
        err = np.zeros((n,), dtype=np.double)
        idC = np.zeros((n,), dtype=np.double)
        idP = np.zeros((n,), dtype=np.intc)
        idI = np.zeros((n,), dtype=np.intc) - 1
        ids = self.i[:, I]
        J = np.lexsort((ids[4, :], ids[3, :], ids[2, :], ids[1, :], ids[0, :]))  # find a better way to sort column
        i0 = I[J[0]]
        num = 0
        for j in J:
            i1 = I[j]
            if np.any(self.i[:, i0] != self.i[:, i1]):
                num += 1
            idP[j] = num
            idC[num] += self.a[i1]
            i0 = i1
            if self.a[i1] > 0:
                idI[num] = i1
        assert(np.all(idC <= c+10**-5))
        assert(np.all(idC >= -10**-5))
        for cnt in range(n):
            i = I[cnt]
            j = idP[cnt]
            self.a[i] = max(min(self.a[i], c), 0)
            ci = max(min(idC[j], c), self.a[i])
            assert(ci <= c+10**-5)
            x1 = self._sparse2dense(self.x[:, i])
            g = np.sum(self.w*x1) - self.b[i]
            pg = g
            if (self.a[i] == 0 and g >= 0) or (ci >= c and g <= 0):
                pg = 0
            if -g > err[j]:
                err[j] = -g
            if self.a[i] == 0 and g > 0:
                self.sv[i] = False
            if ci >= c and g < -10**-12 and self.a[i] < c and idI[j] != i and idI[j] >= 0:
                i2 = idI[j]
                x2 = self._sparse2dense(self.x[:, i2])
                g2 = np.sum(self.w*x2) - self.b[i2]
                numer = g - g2
                if self.a[i] == 0 and numer >0:
                    numer = 0
                    self.sv[i] = False
                if numer > 10**-12 or numer < -10**-12:
                    dA = -numer / (self.d[i] + self.d[i2] - 2 * np.sum(self.x[:, i] * self.x[:, i2]))
                    if dA > 0:
                        dA = min(min(dA, c - self.a[i]), self.a[i2])
                    else :
                        dA = max(max(dA, -self.a[i]), self.a[i2] - c)
                    a1 = self.a[i]
                    a2 = self.a[i2]
                    self.a[i] += dA
                    self.a[i2] -= dA
                    assert(self.a[i]  >= 0 and self.a[i]  <= c)
                    assert(self.a[i2] >= 0 and self.a[i2] <= c)
                    assert(abs(a1 + a2 - (self.a[i] + self.a[i2])) < 10-5)
                    self.w += dA * (x1 - x2)
                    self.project_non_neg()
                    self.l[0] += dA * (self.b[i] - self.b[i2])
            elif pg > 10**-12 or pg < -10**-12:
                dA = self.a[i]
                assert(dA <= ci+10**-5)
                maxA = max(c - ci + dA, 0)
                self.a[i] = min(max(self.a[i] - g/self.d[i], 0), maxA)
                assert(self.a[i] >= 0 and self.a[i] <= c)
                dA = self.a[i] - dA
                self.w += dA*x1
                self.project_non_neg()
                self.l[0] += dA * self.b[i]
                idC[j] = min(max(ci + dA, 0), c)
                assert(idC[j] >= 0 and idC[j] <= c)
            if self.a[i] > 0:
                idI[j] = i
        return np.sum(err)

    def vec2model(self, model, debug=False):
        original_w = self.actual_w().astype(np.double)
        for i in range(np.size(model.defs)):
            x = model.defs[i]
            s = np.shape(x['w'])
            j = range(int(x['i']), int(x['i'] + np.prod(s)))
            model.defs[i]['w'] = np.reshape(original_w[j], s)

        for i in range(np.size(model.filters)):
            x = model.filters[i]
            if x['i'] >= 0:
                s = np.shape(x['w'])
                j = range(int(x['i']), int(x['i'] + np.prod(s)))
                model.filters[i]['w'] = np.reshape(original_w[j], s)
        if True:  # TODO: fix this. w2 is not in the actual form
            self.model2qp(model)
            w_from_updated_model = self.actual_w().astype(np.double)
            if not (np.all(np.ab(original_w - w_from_updated_model) < 10**-5)):
                print np.where(np.ab(original_w - w_from_updated_model) >= 10**-5)
        return model

    def model2qp(self, model):
        w = np.zeros((model.len,))
        w0 = np.zeros_like(w)
        wreg = np.ones_like(w0)
        non_neg = []
        for x in model.defs:
            l = size(x['w'])
            j = np.array(range(x['i'], x['i'] + l))
            w[j] = np.copy(x['w'])
            if l == 1:
                wreg[j] = 0.01
            else:
                wreg[j] = 0.1
                j = np.array([j[0], j[2]])
                w0[j] = 0.01
                non_neg.append(j)
        for x in model.filters:
            if x['i'] >= 0:
                l = size(x['w'])
                j = np.array(range(x['i'], x['i'] + l))
                w[j] = np.copy(x['w'])
        non_neg = np.array(non_neg, dtype=np.intc).flatten() if np.size(non_neg) > 0 else []
        self.w = (w - w0) * wreg
        self.wreg = wreg
        self.w0 = w0
        self.non_neg = non_neg
        return self

    def obtimize(self, model):
        if self.lb < 0 or self.n == np.size(self.a):
            self.mult()
            self.prune()
        else:
            self.one()
        model = self.vec2model(model)
        return model
