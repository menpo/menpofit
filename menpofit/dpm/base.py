import menpo.io as mio
import multiprocessing as mp
import numpy as np
#import scipy.linalg
import scipy.io
import os
from menpo.feature import hog
from menpo.feature.gradient import score
from menpo.image import Image
from numpy import size, nonzero
from scipy.linalg import norm
from menpo.shape import Tree


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

        #self._model_train()

    def _get_config(self):
        # function with configurations for the mixture and number of points.
        conf = {}  # used instead of the opts struct in the original code
        # conf['viewpoint'] = [-30, -15, -15, 0, 15, 15, 30] # range(90, -90, -15)
        # conf['partpoolsize'] =  68 + 68 + 68
        # 39 + 68 + 39  (original code)
        conf['viewpoint'] = range(90, -90, -15)
        conf['partpoolsize'] = 39+68+39

        conf['sbin'] = 4
        conf['mixture_poolid'] = [range(0, 39), range(0, 39), range(39, 107), range(39, 107),
                                  range(39, 107), range(107, 146), range(107, 146)]
        # conf['mixture_poolid'] = [range(0, 68), range(0, 68), range(68, 136), range(68, 136),
        #                           range(68, 136), range(136, 204), range(136, 204)]
        conf['parents'] = [0, 0, 0, 0, 0, 0, 0]  # value to call the get_parents_lns().

        return conf

    def _model_train(self):
        #todo : get this done
        mat_p = '/vol/atlas/homes/grigoris/external/dpm_ramanan/face-release1.0-full/christos_code/'
        file1 = mat_p + 'lfpw_helen_full.mat'
        mat = scipy.io.loadmat(file1)
        multipie = mat['multipie'][0]

        multipiedir = '/vol/hmi/projects/christos/Journal_IVC_2013/01_Data/01_Train/02_LFPW+HELEN/01_Images_Pts/';
        annodir = '/vol/hmi/projects/christos/Journal_IVC_2013/01_Data/01_Train/02_LFPW+HELEN/01_Images_Pts/';
        negims = '/vol/hmi/projects/christos/Journal_IVC_2013/03_Ramanan/INRIA/';

        pickle_dev = '/vol/atlas/homes/ks3811/pickles'
        try:  # TODO: save by model name (otherwise it might load the same data all the time)
            fp = os.path.join(pickle_dev, 'data.pkl')
            _c = mio.import_pickle(fp)
            pos = _c['pos']
            neg = _c['neg']
        except ValueError:  # pickle does not exist
            pos, neg = self._get_image_info(multipie, multipiedir, annodir, negims)
            _c = {'pos': pos}
            _c['neg'] = neg
            mio.export_pickle(_c, fp)

        pos = self._ln2box(pos)
        spos = self._split(pos)
        k = min(len(neg), 200) - 1
        kneg = neg[0:k]

        #todo : come back to make it work in parallel
        #pool = mp.Pool(processes=4)
        #results = pool.map(self._train_model, range(1,7))
        #results = [pool.apply(self._train_model, args=(x,)) for x in range(1, 7)]
        #print results

        parts_models = []
        for i in xrange(self.config['partpoolsize']):
            print i
            init_model = self._init_model(spos[i], self.config['sbin'])
            parts_models.append(self._train(init_model, spos[i], kneg, iter=4))

    def _get_image_info(self, pos_data, pos_data_im, pos_data_anno, neg_data_im):
        # load info for the data.
        # pos_data -> dictionary that includes the image names (for positives).
        # pos_data_im -> dir of positive images
        # pos_data_anno -> dir of annotation of positive images
        # neg_data_im -> dir of negative images
        pos = []
        print('Collecting info for the positive images.')
        for cnt, m in enumerate(pos_data): #     gmixid = cnt
            print cnt
            assert(m['images'].shape[0] > 2)
            for _im in m['images']:
                im_n = _im[0][0]
                ln = mio.import_landmark_file(pos_data_anno + im_n[:im_n.rfind('.') + 1] + 'pts')
                assert (ln.n_landmarks == m['nlandmark'][0][0])

                aux = {}  # aux dictionary, will be saved as element of the 'pos' list.
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
                aux = {}
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
                s['gmixid'] = 1
                s['box_y1'] = p['box_y1'][i] * 1  # idea to implement them as a k*4 numpy array.
                s['box_x1'] = p['box_x1'][i] * 1
                s['box_y2'] = p['box_y2'][i] * 1
                s['box_x2'] = p['box_x2'][i] * 1

                spos[k].append(dict(s))
        return spos

    def _ln2box(self, pos):
        # converts the points in bboxes. (same as point2box in the original code)
        parents_lns = self._get_parents_lns()  # might require modification if more than one models present.
        for i, p in enumerate(pos):
            lengths = np.linalg.norm(p['pts'] - p['pts'][parents_lns, :], axis=1)
            boxlen = np.percentile(lengths, 80.73) / 2.0  # 0.73% to compensate for the zero norm 34 point.
            assert(boxlen > 3)  # ensure that boxes are 'big' enough.

            _t = np.clip(p['pts'] - boxlen, 0, np.inf)
            p['box_y1'] = np.copy(_t[:, 0])
            p['box_x1'] = np.copy(_t[:, 1])
            _t = p['pts'] + boxlen  # no check for boundary
            p['box_y2'] = np.copy(_t[:, 0])
            p['box_x2'] = np.copy(_t[:, 1])

        return pos

    def _get_parents_lns(self, i=0):
        # parents of each landmark point as defined in the original code.
        # i -> indicates the model of landmarks followed. If more than one models,
        # then this could be called with different i value to return the specified parents.
        if i == 0:
            # Point numbering (68 points) should be with the ibug68 convention
            # and expect zero-based numbering (0-67).
            parents_ibug68 = [1, 2, 3, 4, 5, 6, 7, 8, 57, 8, 9, 10, 11, 12, 13, 14,
                              15, 36, 17, 18, 19, 20, 23, 24, 25, 26, 45, 28, 29,
                              30, 33, 32, 33, 33, 33, 34, 37, 38, 39, 27, 39, 40,
                              27, 42, 43, 44, 47, 42, 49, 50, 51, 33, 51, 52, 53,
                              54, 65, 66, 66, 67, 49, 50, 51, 52, 53, 55, 56, 58]
        else:
            raise ValueError('No such model for parents\' list exists.')

        return parents_ibug68

    def _init_model(self, pos_, sbin):
        areas = np.empty((len(pos_),))
        for i, el in enumerate(pos_):
            areas[i] = (el['box_x2'] - el['box_x1'] + 1) * (el['box_y2'] - el['box_y1'] + 1)

        areas = np.sort(areas)
        area = areas[int(areas.shape[0] * 0.2)]
        nw = np.sqrt(area)

        im = hog(Image(np.zeros((1, 30, 30))), mode='sparse', algorithm='zhuramanan')
        size = [im.pixels.shape[0], int(round(nw/sbin)), int(round(nw/sbin))]

        d = {}  # deformation
        d['w'] = 0
        d['i'] = 0
        d['anchor'] = np.zeros((3,))

        f = {}  # filter
        f['w'] = np.empty(size)
        f['i'] = 1

        c1 = {}
        c1['filterid'] = 0
        c1['defid'] = 0
        c1['parent'] = -1
        c = [c1]

        _d, _f, _c = [], [], []  # list of the respective dictionaries above
        _d.append(dict(d)), _f.append(dict(f)), _c.append(c)
        model = {}
        model['defs'] = _d
        model['filters'] = _f
        model['components'] = _c

        model['maxsize'] = size[1:]
        model['len'] = 1 + size[0] * size[1] * size[2]
        model['interval'] = 10
        model['sbin'] = sbin

        model = self._poswarp(model, pos_)
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
        score = np.sum(w * w)
        w2 = w.reshape(s)
        model['filters'][0]['w'] = np.copy(w2)
        model['obj'] = -score * 1.
        return model

    def _warppos(self, model, pos):
        # Load the images, crop and resize them to predefined shape.
        f = model['components'][0][0]['filterid']  # potentially redundant, f == 0 (but check the rest first.)
        siz = model['filters'][f]['w'].shape[1:3]
        sbin = model['sbin']
        pixels = [sbin * siz[0], sbin * siz[1]]
        numpos = len(pos)
        heights = np.empty((numpos,))
        widths = np.empty((numpos,))

        for i, el in enumerate(pos):
            heights[i] = el['box_y2'] - el['box_y1'] + 1
            widths[i] = el['box_x2'] - el['box_x1'] + 1

        cropsize = ((siz[0] + 2) * sbin, (siz[1] + 2) * sbin)

        warped = []
        for i, p in enumerate(pos):

            padx = sbin * widths[i] / pixels[1]
            pady = sbin * heights[i] / pixels[0]

            im = mio.import_image(p['im'])
            im_cr = im.crop([p['box_y1'] - pady, p['box_x1'] - padx], [p['box_y2'] + pady, p['box_x2'] + padx])
            im2 = im_cr.resize(cropsize)
            #warped.append(np.copy(im2.pixels))
            warped.append(im2.copy())

        return warped

    def _train(self, model, pos, neg, iter, c=0.002, wpos=2, wneg=1, maxsize=4, overlap=0.6):
        # the size of vectorized model
        length = self._sparselen(model)
        # Maximum number of model that fitted into the cache
        nmax = round(maxsize*0.25*10**9/length)
        num_id = 5 #5 = [label, id, level, posX, poY]
        qp = Qp(length, nmax, num_id)
        (w, qp.wreg, qp.w0, qp.noneg) = self._model2vec(model)
        qp.cpos = c * wpos
        qp.cneg = c
        qp.w = (w - qp.w0) * qp.wreg
        for t in range(iter):
            model['delta'] = self._poslatent(t, model, qp, pos, overlap)
            if model['delta'] < 0.001:
                break
        return model

    def _sparselen(self, model):
        # check if it can be incorporated to the model (length of filters, deformations)
        len1 = -1
        for tempcnt, comp in enumerate(model['components']):
            numblocks = 0
            feats = np.zeros((model['len'],))
            for p in comp:
                if p == {}:  # dump part for 'root filter'.
                    continue
                x = model['filters'][p['filterid']]
                i1 = x['i']
                i2 = i1 + size(x['w']) - 1
                feats[i1:i2+1] = 1
                numblocks += 1

                x = model['defs'][p['defid']]
                i1 = x['i']
                i2 = i1 + size(x['w']) - 1
                feats[i1:i2+1] = 1
                numblocks += 1
            # Number of entries needed to encode a block-scarce representation
            # 1 maybe used to encode the length itself
            n = 1 + 2 * numblocks + int(np.sum(feats))
            len1 = max(len1, n)
        return len1

    def _model2vec(self, model):
        w = np.zeros((model['len'] + 1,))  # note: +1, otherwise it crashes trying to access the last element. -> check !!!!!!!!!!!!!!!!!!!!
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

        return w, wreg, w0, noneg

    def _poslatent(self, t, model, qp, poses, overlap):
        num_pos = size(poses)
        model['interval'] = 5
        num_positives = np.zeros(size(model['components'], ), dtype=int)
        score0 = qp.score_pos()
        qp.n = 0
        w = (self._model2vec(model)[0] - qp.w0) * qp.wreg
        assert(norm(w - qp.w) < 10**-5)

        for i, pos in enumerate(poses):
            num_parts = size(pos)
            bbox = dict()
            bbox['box'] = np.zeros((num_parts, 4))
            bbox['c'] = pos['gmixid']
            for p in range(num_parts):
                bbox['box'][p, :] = [pos['box_x1'], pos['box_y1'], pos['box_x2'], pos['box_y2']]        #todo : bbox values are weird fixit
            im = mio.import_image(pos['im'])
            im, bbox['box'] = self._croppos(im, bbox['box'])

        return 0

    def _croppos(self, im, box):
        x1 = np.min(box[:, 0])
        y1 = np.min(box[:, 1])
        x2 = np.max(box[:, 2])
        y2 = np.max(box[:, 3])
        print x1, y1, x2, y2
        pad = 0.5 * ((x2 - x1 + 1) + (y2 - y1 + 1))
        print pad
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(im.shape[1] - 1, x2 + pad)
        y2 = min(im.shape[0] - 1, y2 + pad)
        cropped_im = im.crop([y1, x1], [y2, x2])
        print x1, y1, x2, y2
        box[:, 0] -= x1
        box[:, 1] -= y1
        box[:, 2] -= x1
        box[:, 3] -= y1
        print box
        return cropped_im, box


class Qp(object):
    def __init__(self, length, nmax, num_id):
        self.x = np.zeros((length, nmax), dtype=np.float32)
        self.i = np.zeros((num_id, nmax), dtype=np.int)
        self.b = np.zeros((nmax,), dtype=np.float32)
        self.d = np.zeros((nmax,))
        self.a = np.zeros((nmax,))
        self.sv = np.zeros((nmax,), dtype=np.bool)
        self.w = np.zeros((length,))
        self.l = 0
        self.n = 0
        self.ub = 0.0
        self.lb = 0.0
        self.svfix = []

    def score_pos(self):
        y = self.i[0, 0:self.n]
        i = np.array(nonzero(y > 1)[0], dtype=np.intc)
        w = self.w + self.w0 * self.wreg
        scores = score(w, self.x, i)/self.cpos
        return scores




