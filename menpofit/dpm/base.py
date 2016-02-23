import menpo.io as mio
import numpy as np
import scipy.io
import time
import os
from menpo.feature import hog
from menpo.feature.gradient import score, lincomb, qp_one_sparse
from menpo.image import Image
from .fitter import DPMFitter, non_max_suppression_fast, clip_boxes, bb_to_lns
from numpy import size, nonzero
from menpo.shape import Tree
from scipy.sparse import csr_matrix


class DPM(object):
    def __init__(self, filters_all, defs_all, components=None, config=None):
        # assert(isinstance(tree, Tree))
        # assert(len(filters) == len(tree.vertices) == len(def_coef))
        self.components = components
        self.filters_all = filters_all
        self.defs_all = defs_all
        self.config = config

        # remove when config is passed properly or can be learned from the example.
        if self.config is None:
            self.config = self._get_config()

        self.fitter = DPMFitter(self)
        # self._model_train()

    @staticmethod
    def _get_config():
        # function with configurations for the mixture and number of points.
        conf = dict()  # used instead of the opts struct in the original code
        # conf['viewpoint'] = [-30, -15, -15, 0, 15, 15, 30] # range(90, -90, -15)
        # conf['partpoolsize'] =  68 + 68 + 68
        # 39 + 68 + 39  (original code)
        conf['viewpoint'] = range(90, -90, -15)
        conf['partpoolsize'] = 39+68+39

        conf['sbin'] = 4
        conf['mixture_poolid'] = [range(0, 39), range(0, 39), range(0, 39),
                                  range(0, 68), range(0, 68), range(0, 68), range(0, 68),
                                  range(0, 68), range(0, 68), range(0, 68),
                                  # range(39, 107), range(39, 107), range(39, 107), range(39, 107),
                                  # range(39, 107), range(39, 107), range(39, 107),
                                  range(107, 146), range(107, 146), range(107, 146)]
        # conf['mixture_poolid'] = [range(0, 68), range(0, 68), range(68, 136), range(68, 136),
        #                           range(68, 136), range(136, 204), range(136, 204)]
        conf['parents'] = [0, 0, 0, 0, 0, 0, 0]  # value to call the get_parents_lns().

        return conf

    @staticmethod
    def get_pos(self):
        pickle_dev = '/vol/atlas/homes/ks3811/pickles'
        fp = os.path.join(pickle_dev, 'data.pkl')
        _c = mio.import_pickle(fp)
        pos = _c['pos']
        return pos

    def _model_train(self):
        # todo : get this done
        # mat_p = '/vol/atlas/homes/grigoris/external/dpm_ramanan/face-release1.0-full/christos_code/'
        # file1 = mat_p + 'lfpw_helen_full.mat'
        file1 = '/homes/ks3811/Phd/1 year/wildFace/face-release1.0-full/multipie.mat'
        mat = scipy.io.loadmat(file1)
        multipie = mat['multipie'][0]

        # multipiedir = '/vol/hmi/projects/christos/Journal_IVC_2013/01_Data/01_Train/02_LFPW+HELEN/01_Images_Pts/';
        multipiedir = '/vol/hmi/projects/christos/Journal_IVC_2013/01_Data/01_Train/01_MultiPIE/01_Images_Pts/';
        # multipiedir = '/vol/atlas/databases/multipie/';
        # annodir = '/vol/hmi/projects/christos/Journal_IVC_2013/01_Data/01_Train/02_LFPW+HELEN/01_Images_Pts/';
        # annodir = '/vol/hmi/projects/christos/Journal_IVC_2013/01_Data/01_Train/01_MultiPIE/01_Images_Pts/';
        annodir = '/homes/ks3811/Phd/1 year/wildFace/face-release1.0-full/annos.mat';
        negims = '/vol/hmi/projects/christos/Journal_IVC_2013/03_Ramanan/INRIA/';

        pickle_dev = '/vol/atlas/homes/ks3811/pickles'
        try:  # TODO: save by model name (otherwise it might load the same data all the time)
            fp = os.path.join(pickle_dev, 'data2.pkl')
            _c = mio.import_pickle(fp)
            pos = _c['pos']
            neg = _c['neg']
        except ValueError:  # pickle does not exist
            pos, neg = self._get_pie_image_info(multipie, multipiedir, annodir, negims)
            _c = dict()
            _c['pos'] = pos
            _c['neg'] = neg
            mio.export_pickle(_c, fp)

        pos = self._ln2box(pos)
        spos = self._split(pos)
        k = min(len(neg), 200)
        kneg = neg[0:k]

        #todo : come back to make it work in parallel
        #pool = mp.Pool(processes=4)
        #results = pool.map(self._train_model, range(1,7))
        #results = [pool.apply(self._train_model, args=(x,)) for x in range(1, 7)]
        #print results

        parts_models = []
        try:
            fp = os.path.join(pickle_dev, 'actual_parts_model_fast.pkl')
            parts_models = mio.import_pickle(fp)
        except ValueError:  # pickle does not exist
            import time
            start = time.time()
            for i in xrange(self.config['partpoolsize']):
                if len(spos[i]) > 0:
                    init_model = self._init_model(spos[i], self.config['sbin'])
                    parts_models.append(self._train(init_model, spos[i], kneg, iters=4))
                else:
                    parts_models.append([])
            end = time.time()
            print (end - start)
            fp = os.path.join(pickle_dev, 'actual_parts_model_fast.pkl')
            mio.export_pickle(parts_models, fp)

        try:
            fp = os.path.join(pickle_dev, 'defs.pkl')
            defs = mio.import_pickle(fp)
        except ValueError:  # pickle does not exist
            defs = self.build_mixture_defs(pos, parts_models[39]['maxsize'][0])
            fp = os.path.join(pickle_dev, 'defs.pkl')
            mio.export_pickle(defs, fp)

        try:
            fp = os.path.join(pickle_dev, 'mix2.pkl')
            model = mio.import_pickle(fp)
        except ValueError:  # pickle does not exist
            import time
            start = time.time()
            model = self.build_mixture_model(parts_models, defs)
            model = self._train(model, pos, kneg, 1)
            end = time.time()
            print (end - start)
            fp = os.path.join(pickle_dev, 'mix2.pkl')
            mio.export_pickle(model, fp)

        try:
            fp = os.path.join(pickle_dev, 'final.pkl')
            model = mio.import_pickle(fp)
        except ValueError:  # pickle does not exist
            import time
            start = time.time()
            model = self._train(model, pos, neg, 2)
            end = time.time()
            print (end - start)
            fp = os.path.join(pickle_dev, 'final.pkl')
            mio.export_pickle(model, fp)
            print 'done'



    @staticmethod
    def _get_image_info(pos_data, pos_data_im, pos_data_anno, neg_data_im):
        # load info for the data.
        # pos_data -> dictionary that includes the image names (for positives).
        # pos_data_im -> dir of positive images
        # pos_data_anno -> dir of annotation of positive images
        # neg_data_im -> dir of negative images
        pos = []
        print('Collecting info for the positive images.')
        for cnt, m in enumerate(pos_data):  # gmixid = cnt
            print cnt
            assert(m['images'].shape[0] > 2)
            for _im in m['images']:
                im_n = _im[0][0]
                ln = mio.import_landmark_file(pos_data_anno + im_n[:im_n.rfind('.') + 1] + 'pts')
                assert (ln.n_landmarks == m['nlandmark'][0][0])

                aux = dict()  # aux dictionary, will be saved as element of the 'pos' list.
                aux['pts'] = ln.lms.points.copy()   # Differs from original code with the tree!!!
                aux['im'] = os.path.join(pos_data_im, im_n)
                assert(os.path.isfile(aux['im']))
                aux['gmixid'] = 1 * cnt

                pos.append(dict(aux))

        print('Collecting info for the negative images.')
        l1 = sorted(os.listdir(neg_data_im))
        neg = []
        for elem in l1:
            print elem
            if elem[elem.rfind('.') + 1:] in ['jpg', 'png', 'jpeg']:
                aux = dict()
                aux['im'] = os.path.join(neg_data_im, elem)
                neg.append(dict(aux))

        return pos, neg

    @staticmethod
    def _get_pie_image_info(pos_data, pos_data_im, pos_data_anno, neg_data_im):
        pos = []
        train_list = [50, 50, 50, 50, 50, 50, 300, 50, 50, 50, 50, 50, 50]
        mat = scipy.io.loadmat(pos_data_anno)
        annos = mat['annos'][0, 0]
        for cnt, m in enumerate(pos_data): #     gmixid = cnt
            assert(m['images'].shape[0] > 2)
            count = 0
            for _im in m['images']:
                im_n = _im[0][0]
                # anno_file_name = pos_data_anno + im_n + '.pts'
                file_name = pos_data_im + im_n + '.png'
                if count > train_list[cnt]:
                    break
                count += 1
                if not os.path.isfile(file_name):  # or not os.path.isfile(anno_file_name)):
                    continue
                # ln = mio.import_landmark_file(anno_file_name)
                # assert (ln.n_landmarks == m['nlandmark'][0][0])

                aux = dict()  # aux dictionary, will be saved as element of the 'pos' list.
                aux['pts'] = annos['n' + im_n]  # ln.lms.points.copy()   # Differs from original code with the tree!!!
                aux['im'] = file_name  # os.path.join(pos_data_im, im_n)
                assert(os.path.isfile(aux['im']))
                aux['gmixid'] = 1 * cnt
                pos.append(dict(aux))

        print('Collecting info for the negative images.')
        l1 = sorted(os.listdir(neg_data_im))
        neg = []
        for elem in l1:
            if elem[elem.rfind('.') + 1:] in ['jpg', 'png', 'jpeg']:
                aux = dict()
                aux['im'] = os.path.join(neg_data_im, elem)
                neg.append(dict(aux))
        return pos, neg

    def _split(self, pos_images):
        # split the boxes into different splits. Each box (formed from the respective landmark) is assigned to
        # the classes it belongs according to the config file, e.g. for 68 lns -> 68 splits.
        conf = self.config
        spos = []
        for i in range(conf['partpoolsize']):
            spos.append([])

        for p in pos_images:
            partids_inpool = conf['mixture_poolid'][p['gmixid']]
            for i, k in enumerate(partids_inpool):
                s = {}
                s['im'] = p['im'][:]  # copy the string
                s['gmixid'] = 0
                s['box_y1'] = [p['box_y1'][i] * 1]  # idea to implement them as a k*4 numpy array.
                s['box_x1'] = [p['box_x1'][i] * 1]
                s['box_y2'] = [p['box_y2'][i] * 1]
                s['box_x2'] = [p['box_x2'][i] * 1]

                spos[k].append(dict(s))
        return spos

    def _ln2box(self, pos):
        # converts the points in bboxes. (same as point2box in the original code)
        for i, p in enumerate(pos):
            parents_lns = self._get_parents_lns(p['gmixid'])  # might require modification if more than one models present.
            lengths = np.linalg.norm(abs(p['pts'][1:] - p['pts'][parents_lns[0, 1:], :]), axis=1)
            boxlen = np.percentile(lengths, 80)/2  # 0.73% to compensate for the zero norm 34 point.
            assert(boxlen > 3)  # ensure that boxes are 'big' enough.
            _t = np.clip(p['pts'] - boxlen - 1, 0, np.inf) # -1 for matlab indexes
            p['box_x1'] = np.copy(_t[:, 0])
            p['box_y1'] = np.copy(_t[:, 1])
            _t = p['pts'] - 1 + boxlen  # no check for boundary, -1 for matlab indexes
            p['box_x2'] = np.copy(_t[:, 0])
            p['box_y2'] = np.copy(_t[:, 1])
        return pos

    @staticmethod
    def _get_parents_lns(gmixid):
        # parents of each landmark point as defined in the original code.
        # i -> indicates the model of landmarks followed. If more than one models,
        # then this could be called with different i value to return the specified parents.
        if 0 <= gmixid <= 2:
            parents = np.array(([0, 1, 2, 3, 4, 5,
                            1, 7, 8, 9, 10,
                            11, 12, 13, 14,
                            1, 16, 17, 18, 19, 20, 21,
                            19, 23, 24, 23, 26,
                            22, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]))
            # Point numbering (68 points) should be with the ibug68 convention
            # and expect zero-based numbering (0-67).
            # parents_ibug68 = [1, 2, 3, 4, 5, 6, 7, 8, 57, 8, 9, 10, 11, 12, 13, 14,
            #                   15, 36, 17, 18, 19, 20, 23, 24, 25, 26, 45, 28, 29,
            #                   30, 33, 32, 33, 33, 33, 34, 37, 38, 39, 27, 39, 40,
            #                   27, 42, 43, 44, 47, 42, 49, 50, 51, 33, 51, 52, 53,
            #                   54, 65, 66, 66, 67, 49, 50, 51, 52, 53, 55, 56, 58]
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
                       51, 52, 53, 54, 55, 56, 57, 58, 59, 52, 61, 62, 63, 64, 65, 66, 67],))
        elif 10 <= gmixid <= 12:
            parents = np.array(([0, 1, 2, 3, 4, 5,
                                 1, 7, 8, 9, 10,
                                 11, 12, 13, 14,
                                 1, 16, 17, 18, 19, 20, 21,
                                 19, 23, 24, 23, 26,
                                 22, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]))
        else:
            raise ValueError('No such model for parents\' list exists.')
        return parents - 1  # parents_ibug68

    def _init_model(self, pos_, sbin):
        areas = np.empty((len(pos_),))
        for i, el in enumerate(pos_):
            areas[i] = (el['box_x2'][0] - el['box_x1'][0] + 1) * (el['box_y2'][0] - el['box_y1'][0] + 1)

        areas = np.sort(areas)
        area = areas[int(areas.shape[0] * 0.3)]  # todo: return to 0.2. Current value of 0.21 just to make sz 5 and easier to work with
        nw = np.sqrt(area)

        im = hog(Image(np.zeros((1, 30, 30))), mode='sparse', algorithm='zhuramanan')
        siz = [im.pixels.shape[0], round(nw/sbin), round(nw/sbin)]

        d = dict()  # deformation
        d['w'] = 0
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
        c['tree'] = Tree(np.array([0]), root_vertex=0, skip_checks=True)

        _d, _f, _c = [], [], []  # list of the respective dictionaries above
        _d.append(dict(d)), _f.append(dict(f)), _c.append(c)
        model = dict()
        model['defs'] = _d
        model['filters'] = _f
        model['components'] = _c

        model['maxsize'] = siz[1:]
        model['len'] = 1 + siz[0] * siz[1] * siz[2]
        model['interval'] = 10
        model['sbin'] = sbin

        model = self._poswarp(model, pos_)
        c = model['components'][0]
        c['bias'] = model['defs'][0]['w']
        return model

    def _poswarp(self, model, pos):
        warped = self._warppos(model, pos[0:100])  ################## DEBUGGING, REMOVE THE 0:100 in the end!!!!!!!!!!!!!
        s = model['filters'][0]['w'].shape  # filter size
        num = len(warped)
        feats = np.empty((s[0]*s[1]*s[2], num))

        for c, im in enumerate(warped):
            feat = hog(im, mode='sparse', algorithm='zhuramanan', cell_size=model['sbin'])
            feats[:, c] = feat.pixels.flatten()

        w = np.mean(feats, 1)
        scores = np.sum(w * w)
        w2 = w.reshape(s)
        model['filters'][0]['w'] = np.copy(w2)
        model['obj'] = - scores * 1.
        return model

    def _warppos(self, model, pos):
        # Load the images, crop and resize them to predefined shape.
        f = model['components'][0]['filter_ids'][0]  # potentially redundant, f == 0 (but check the rest first.)
        siz = model['filters'][f]['w'].shape[1:3]
        sbin = model['sbin']
        pixels = [sbin * siz[0], sbin * siz[1]]
        numpos = len(pos)
        heights = np.empty((numpos,))
        widths = np.empty((numpos,))

        for i, el in enumerate(pos):
            heights[i] = el['box_y2'][0] - el['box_y1'][0] + 1
            widths[i] = el['box_x2'][0] - el['box_x1'][0] + 1

        cropsize = ((siz[0] + 2) * sbin, (siz[1] + 2) * sbin)

        warped = []
        for i, p in enumerate(pos):

            padx = sbin * widths[i] / pixels[1]
            pady = sbin * heights[i] / pixels[0]

            im = mio.import_image(p['im'])
            im_cr = im.crop([p['box_y1'][0] - pady, p['box_x1'][0] - padx], [p['box_y2'][0] + pady, p['box_x2'][0] + padx])
            im2 = im_cr.resize(cropsize)
            #warped.append(np.copy(im2.pixels))
            warped.append(im2.copy())

        return warped

    def _train(self, model, pos, negs, iters, c=0.002, wpos=2, wneg=1, maxsize=4, overlap=0.6):
        # the size of vectorized model
        length = self._sparselen(model)
        # Maximum number of model that fitted into the cache
        nmax = round(maxsize*0.25*10**9/length)
        num_id = 5  # 5 = [label, id, level, posX, poY]
        qp = Qp(length, nmax, num_id, c, wpos)
        (w, qp.wreg, qp.w0, qp.noneg) = self._model2vec(model)
        qp.w = (w - qp.w0) * qp.wreg
        for t in range(iters):
            model['delta'] = self._poslatent(t, model, qp, pos, overlap)
            if model['delta'] < 0.001:
                break
            qp.svfix = range(qp.n)
            qp.sv[qp.svfix] = True
            qp.prune()
            qp.opt(0.5)
            model = self.vec2model(qp.actual_w(), model)
            model['intervel'] = 4

            for i, neg in enumerate(negs):
                print 'iter:', t, 'neg:', i, '/', np.size(negs)
                im = mio.import_image(neg['im'])
                box, model = DPMFitter(self).fit_with_bb(im,  model, -1, [], 0, i, -1, qp)
                if np.sum(qp.sv) == nmax:
                    break

            qp.opt()
            model = self.vec2model(qp.actual_w(), model)
            model['lb'] = qp.lb
            model['ub'] = qp.ub

        return model

    def _sparselen(self, model):
        # check if it can be incorporated to the model (length of filters, deformations)
        len1 = -1
        for tempcnt, comp in enumerate(model['components']):
            if not comp:
                continue
            numblocks = 0
            feats = np.zeros((model['len'],))
            for cv in comp['tree'].vertices:
                x = model['filters'][comp['filter_ids'][cv]]
                i1 = x['i']
                i2 = i1 + size(x['w']) - 1
                feats[i1:i2+1] = 1
                numblocks += 1

                x = model['defs'][comp['def_ids'][cv]]
                i1 = x['i']
                i2 = i1 + size(x['w']) - 1
                feats[i1:i2+1] = 1
                numblocks += 1
            # Number of entries needed to encode a block-scarce representation
            # 1 maybe used to encode the length itself
            n = 1 + 2 * numblocks + int(np.sum(feats))
            len1 = max(len1, n)
        return len1

    @staticmethod
    def _model2vec(model):
        w = np.zeros((model['len'],))  # note: +1, otherwise it crashes trying to access the last element. -> check !!!!!!!!!!!!!!!!!!!!
        w0 = np.zeros_like(w)
        wreg = np.ones_like(w0)
        noneg = []

        for x in model['defs']:
            l = size(x['w'])
            j = np.array(range(x['i'], x['i'] + l))
            w[j] = np.copy(x['w'])

            if l == 1:
                wreg[j] = 0.01
            else:
                wreg[j] = 0.1
                j = np.array([j[0], j[2]])
                w0[j] = 0.01
                noneg.append(j)

        for x in model['filters']:
            if x['i'] >= 0:
                l = size(x['w'])
                j = np.array(range(x['i'], x['i'] + l))
                w[j] = np.copy(x['w'])

        noneg = np.array(noneg, dtype=np.intc).flatten() if np.size(noneg) > 0 else []  # np.zeros((1, 1), dtype=np.intc)

        return w, wreg, w0, noneg

    def _poslatent(self, t, model, qp, poses, overlap):
        num_pos = size(poses)
        model['interval'] = 5
        num_positives = np.zeros(size(model['components'], ), dtype=int)
        score0 = qp.score_pos()
        qp.n = 0
        w = (self._model2vec(model)[0] - qp.w0) * qp.wreg
        assert(scipy.linalg.norm(w - qp.w) < 10**-5)

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
            start = time.time()
            box, model = DPMFitter(self).fit_with_bb(im,  model, -np.inf, bbox, overlap, i, 1, qp)
            end = time.time()
            # print 'taken:', end - start
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

    def vec2model(self, w, model):
        w = w.astype(np.double)
        for i in range(np.size(model['defs'])):
            x = model['defs'][i]
            s = np.shape(x['w'])
            j = range(int(x['i']), int(x['i'] + np.prod(s)))
            model['defs'][i]['w'] = np.reshape(w[j], s)

        for i in range(np.size(model['filters'])):
            x = model['filters'][i]
            if x['i'] >= 0:
                s = np.shape(x['w'])
                j = range(int(x['i']), int(x['i'] + np.prod(s)))
                model['filters'][i]['w'] = np.reshape(w[j], s)
        w2 = self._model2vec(model)
        assert(np.all(w == w2[0]))
        return model

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

            deftemp = (points[:, :, :] - points[par, :, :])[0]
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
        model_['maxsize'] = models[39]['maxsize']
        model_['len'] = 0
        model_['interval'] = models[39]['interval']
        model_['sbin'] = models[39]['sbin']

        # add the filters
        model_['filters'] = []
        for m in models:    # for each possible parts
            if np.size(m) > 0:
                f = dict()
                f['w'] = m['filters'][0]['w']
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
                # d['w'] = np.array([0.00, 0, 0.00, 0])
                d['w'] = np.array([0.01, 0, 0.01, 0])
                d['anchor'] = np.array([round(def_j[0]) + 1, round(def_j[1]) + 1, 0])
                model_['defs'].append(d)
                model_['len'] += np.size(d['w'])
                def_ids.append(nd)
            component['def_ids'] = def_ids
            component['filter_ids'] = conf['mixture_poolid'][i]
            parents = self._get_parents_lns(i)[0]
            num_parts = np.size(parents)
            pairs = zip(parents, range(num_parts))
            tree_matrix = csr_matrix(([1] * (num_parts-1), (zip(*pairs[1:]))), shape=(num_parts, num_parts))
            component['tree'] = Tree(tree_matrix, root_vertex=0, skip_checks=True)
            model_['components'].append(component)
        return model_

class Qp(object):
    def __init__(self, length, nmax, num_id, c, wpos):
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
        self.svfix = []
        self.cpos = c * wpos
        self.cneg = c

    def score_pos(self):
        y = self.i[0, 0:self.n]
        i = np.array(nonzero(y)[0], dtype=np.intc)
        w = self.w + self.w0 * self.wreg
        scores = score(w, self.x, i)/self.cpos
        # assert(scores[-1] == self.slow_score(w/self.cpos, self.x[:, self.n-1]))
        return scores

    def score_neg(self):
        w = - (self.w + self.w0 * self.wreg)
        scores = score(w, self.x, np.array([self.n - 1],  dtype=np.intc)) / self.cneg  # -1 bc current n point to the next free space
        # assert(scores == self.slow_score(w, self.x[:, self.n-1])/self.cneg)
        return scores

    @staticmethod
    def slow_score(w, x):
        xp = 1
        y = 0
        scores = []
        for b in range(x[0]):
            wp = int(x[xp])
            xp += 1
            length = int(x[xp]) - wp
            xp += 1
            # print 'wp :', wp, 'length :', length
            score = 0
            for j in range(length):
                if length == 775:
                    score += w[wp] * x[xp]
                wp += 1
                xp += 1
            if length == 775:
                scores.append(score)
            y += score
            # print 'current_score :', score, 'acc_score :', y
        return scores

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
            # if np.size(x) > 1:
            #     actual_x = np.zeros((np.shape(x)[0], np.shape(x)[2], np.shape(x)[1]))
            #     for f in range(np.shape(x)[0]):
            #         actual_x[f, :, :] = np.transpose(np.asmatrix(x[f, :, :]))
            #     print np.all(x == actual_x)
            #     x = actual_x
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
        self.project_noneg()
        self.lb = self.update_lb()
        self.n = n

    def update_lb(self):
        return self.l[0] - 0.5 * np.sum(self.w * self.w)

    def update_ub(self, loss):
        return loss + 0.5 * np.sum(self.w * self.w)

    def increment_neg_ub(self, rscore):
        self.ub += self.cneg*max(1+rscore, 0)

    def project_noneg(self):
        if np.size(self.noneg) > 0:
            self.w[self.noneg] = np.maximum(self.w[self.noneg], np.zeros_like(self.w[self.noneg]))

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
        J = np.lexsort((ids[4, :], ids[3, :], ids[2, :], ids[1, :], ids[0, :]))  # find a better way to sort column
        ids = ids[:, J]
        eqid = np.zeros_like(J, dtype=bool)
        eqid[1:] = np.all(ids[:, 1:] == ids[:, 0:-1], axis=0)
        slack = self.b[I] - score(self.w, self.x, I)
        loss = self.computeloss(slack[J], eqid)
        ub = self.update_ub(loss)
        lb = self.lb
        # print 'lb : ', lb, 'ub : ', ub
        self.sv[I] = True
        for t in range(iters):
            self.one()
            lb = self.lb
            ub_est = min(self.ub, ub)
            # print 'lb : ', lb, 'est_ub : ', ub_est
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
        #self.lin_comb(idxs)
        self.project_noneg()
        self.lb_old = self.lb
        self.lb = self.update_lb()
        if self.lb_old != 0:
            assert(self.lb > self.lb_old - 10**-5)

    def lin_comb(self, ids):
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
        assert(np.all(self.w[self.noneg] >= 0))
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
                    self.project_noneg()
                    self.l[0] += dA * (self.b[i] - self.b[i2])
            elif pg > 10**-12 or pg < -10**-12:
                dA = self.a[i]
                assert(dA <= ci+10**-5)
                maxA = max(c - ci + dA, 0)
                self.a[i] = min(max(self.a[i] - g/self.d[i], 0), maxA)
                assert(self.a[i] >= 0 and self.a[i] <= c)
                dA = self.a[i] - dA
                self.w += dA*x1
                self.project_noneg()
                self.l[0] += dA * self.b[i]
                idC[j] = min(max(ci + dA, 0), c)
                assert(idC[j] >= 0 and idC[j] <= c)
            if self.a[i] > 0:
                idI[j] = i
        return np.sum(err)
