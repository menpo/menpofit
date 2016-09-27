from __future__ import print_function
import time
import os
from math import log as log_m
import numpy as np
from scipy.sparse import csr_matrix
import scipy.io

import menpo.io as mio
from menpo.feature import hog
from menpo.image import Image
from menpo.shape import Tree, PointCloud

from .fitter import DPMFitter, non_max_suppression_fast, clip_boxes, bb_to_lns
from .utils import score, lincomb, qp_one_sparse


class DPMLearner(object):
    def __init__(self, config=None, feature_pyramid=None):
        self.config = config
        self.feature_pyramid = feature_pyramid

        if self.config is None:
            self.config = self._get_face_default_config()

        if self.feature_pyramid is None:
            self.feature_pyramid = HogFeaturePyramid()

    @staticmethod
    def _get_face_default_config():
        side_faces_parts = 39  # 39 parts for side faces.
        frontal_faces_parts = 68  # 68 parts for near frontal faces.
        conf = dict()  # configurations for the mixture components.
        conf['viewpoint'] = range(90, -90-15, -15)  # 90 <-> -90
        conf['partpoolsize'] = side_faces_parts + frontal_faces_parts + side_faces_parts
        right_side_face_indexes = range(0, side_faces_parts)
        frontal_face_indexex = range(side_faces_parts, side_faces_parts + frontal_faces_parts)
        left_side_face_indexes = range(side_faces_parts + frontal_faces_parts, frontal_faces_parts + 2*side_faces_parts)
        conf['mixture_poolid'] = [right_side_face_indexes, right_side_face_indexes, right_side_face_indexes,
                                  frontal_face_indexex, frontal_face_indexex, frontal_face_indexex,
                                  frontal_face_indexex, frontal_face_indexex, frontal_face_indexex,
                                  frontal_face_indexex, left_side_face_indexes, left_side_face_indexes,
                                  left_side_face_indexes]
        conf['poolid'] = list()
        count = 0
        for i in range(np.size(conf['viewpoint'])):
            mixture_part_number = np.size(conf['mixture_poolid'][i])
            mixture = np.array(range(0, mixture_part_number)) + count
            conf['poolid'].append(mixture)
            count += mixture_part_number
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
        return parents - 1  # -1 to change matlab indexes into python indexes

    @staticmethod
    def _get_anno2tree(gmixid):
        # convert original annotations to match tree structure
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
    def _get_pie_image_info(pickle_dev):
        multipie_mat = '/vol/atlas/homes/ks3811/matlab/multipie.mat'
        pos_data = scipy.io.loadmat(multipie_mat)['multipie'][0]

        pos_data_dir = '/vol/hci2/Databases/video/MultiPIE'
        anno_dir = '/vol/atlas/homes/ks3811/matlab/my_annotation'

        neg_data_dir = '/vol/atlas/homes/ks3811/matlab/INRIA'

        try:  # Check if the data is already existed.
            fp = os.path.join(pickle_dev, 'data.pkl')
            _c = mio.import_pickle(fp)
            pos = _c['pos']
            neg = _c['neg']
        except ValueError:  # Data in pickle does not exist
            start = time.time()
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
                    file_name = '{0}/session{1}/png/{2}/{3}/{4}_{5}/{6}.png'.format(pos_data_dir, ses_id, sub_id,
                                                                                    rec_id, cam1_id, cam2_id, img_name)
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
            _c = dict()
            _c['pos'] = pos
            _c['neg'] = neg
            mio.export_pickle(_c, fp)
            stop = time.time()
            fp = os.path.join(pickle_dev, 'data_time.pkl')
            mio.export_pickle(stop-start, fp)
        return pos, neg

    @staticmethod  # this method might not be needed anymore.
    def _get_jpg_pie_image_info(pickle_dev):
        pickle_dev = '/vol/atlas/homes/ks3811/pickles/data'
        multipie_mat = '/vol/atlas/homes/ks3811/matlab/multipie.mat'
        pos_data = scipy.io.loadmat(multipie_mat)['multipie'][0]
        anno_dir = '/vol/atlas/homes/ks3811/matlab/my_annotation'
        neg_data_dir = '/vol/atlas/homes/ks3811/matlab/INRIA'

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
                jpg_filename = '{0}/pos/{1}_{2}_{3}_{4}_{5}_{6}.jpg'.format(pickle_dev, ses_id, sub_id, rec_id,
                                                                            cam1_id, cam2_id, img_name)
                anno_file_name = '{0}/{1}_lm.mat'.format(anno_dir, img_name)
                if not os.path.isfile(jpg_filename) or not os.path.isfile(anno_file_name):
                    continue
                count += 1
                aux = dict()
                aux['pts'] = scipy.io.loadmat(anno_file_name)['pts'][DPMLearner._get_anno2tree(gmixid), :]
                aux['im'] = jpg_filename
                aux['gmixid'] = gmixid
                pos.append(aux)
        print('Collecting info for the negative images.')
        l1 = sorted(os.listdir(neg_data_dir))
        neg = []
        for elem in l1:
            if elem[elem.rfind('.') + 1:] in ['jpg', 'png', 'jpeg']:
                aux = dict()
                jpg_file_name = '{0}/neg/{1}.jpg'.format(pickle_dev, elem[:-(len(elem) - elem.rfind('.'))])
                aux['im'] = jpg_file_name
                neg.append(dict(aux))
        _c = dict()
        _c['pos'] = pos
        _c['neg'] = neg
        return pos, neg

    def _model_train(self, pickle_dev, small_k=200):
        pos, neg = self._get_pie_image_info(pickle_dev)

        pos = self._ln2box(pos)
        spos = self._split(pos)
        k = min(len(neg), small_k)
        kneg = neg[0:k]

        file_name = 'actual_parts_model_fast'
        parts_models_file_name, parts_models_time_file_name = self._get_files_names(file_name)
        parts_models = []
        try:
            fp = os.path.join(pickle_dev, parts_models_file_name)
            parts_models = mio.import_pickle(fp)
        except ValueError:
            start = time.time()
            for i in xrange(self.config['partpoolsize']):
                assert(len(spos[i]) > 0)
                init_model = self._init_model(spos[i])
                parts_models.append(self._train(init_model, spos[i], kneg, iters=4))
            fp = os.path.join(pickle_dev,  parts_models_file_name)
            mio.export_pickle(parts_models, fp)
            stop = time.time()
            fp = os.path.join(pickle_dev, parts_models_time_file_name)
            mio.export_pickle(stop-start, fp)

        file_name = 'defs'
        defs_file_name, _ = self._get_files_names(file_name)
        try:  # todo: if independent parts are learned in parallel, need to wait for all results before continuing.
            fp = os.path.join(pickle_dev, defs_file_name)
            defs = mio.import_pickle(fp)
        except ValueError:
            defs = self.build_mixture_defs(pos, parts_models[0].maxsize[0])
            fp = os.path.join(pickle_dev, defs_file_name)
            mio.export_pickle(defs, fp)

        file_name = 'mix'
        mix_file_name, mix_time_file_name = self._get_files_names(file_name)
        try:
            fp = os.path.join(pickle_dev, mix_file_name)
            model = mio.import_pickle(fp)
        except ValueError:
            model = self.build_mixture_model(parts_models, defs)
            start = time.time()
            model = self._train(model, pos, kneg, 1)
            fp = os.path.join(pickle_dev, mix_file_name)
            mio.export_pickle(model, fp)
            stop = time.time()
            fp = os.path.join(pickle_dev, mix_time_file_name)
            mio.export_pickle(stop-start, fp)

        file_name = 'final2'
        final_file_name, final_time_file_name = self._get_files_names(file_name)
        try:
            fp = os.path.join(pickle_dev, final_file_name)
            model = mio.import_pickle(fp)
        except ValueError:
            start = time.time()
            model = self._train(model, pos, neg, 2)
            fp = os.path.join(pickle_dev, final_file_name)
            mio.export_pickle(model, fp)
            stop = time.time()
            fp = os.path.join(pickle_dev, final_time_file_name)
            mio.export_pickle(stop-start, fp)

    @staticmethod
    def _get_files_names(file_name):
        dot_file_name = '.pkl'
        time_str = '_time'
        time_file_name = file_name + time_str + dot_file_name
        file_name += dot_file_name
        return file_name, time_file_name

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
        # Converts the points into bounding boxes.
        for i, p in enumerate(pos):
            parents_lns = self._get_parents_lns(p['gmixid'])
            lengths = np.linalg.norm(abs(p['pts'][1:] - p['pts'][parents_lns[1:]]), axis=1)
            boxlen = np.percentile(lengths, 80, interpolation='midpoint')
            assert(boxlen > 3)  # ensure that boxes are 'big' enough.
            _t = np.clip(p['pts'] - 1 - boxlen/2, 0, np.inf) # -1 for matlab indexes
            p['box_x1'] = np.copy(_t[:, 0])
            p['box_y1'] = np.copy(_t[:, 1])
            _t = p['pts'] - 1 + boxlen/2  # no check for boundary, -1 for matlab indexes
            p['box_x2'] = np.copy(_t[:, 0])
            p['box_y2'] = np.copy(_t[:, 1])
        return pos

    def _init_model(self, pos_):
        areas = np.empty((len(pos_),))
        for i, el in enumerate(pos_):
            areas[i] = (el['box_x2'][0] - el['box_x1'][0] + 1) * (el['box_y2'][0] - el['box_y1'][0] + 1)
        areas = np.sort(areas)
        area = areas[int(np.floor(areas.size * 0.2))]  # Pick the 20th percentile area
        nw = np.sqrt(area)

        spacial_scale = self.feature_pyramid.spacial_scale
        siz = [self.feature_pyramid.feature_size, 5, 5]  # int(round(nw/spacial_scale)), int(round(nw/spacial_scale))]
        # todo: see how this can be solved instead of using const [5, 5]

        d = dict()  # deformation
        d['w'] = 0  # bias
        d['i'] = 0
        d['anchor'] = np.zeros((3,))

        f = dict()  # filter
        f['w'] = np.random.rand(self.feature_pyramid.feature_size, 5, 5)  # np.empty(siz)
        f['i'] = 1

        c = dict()
        c['filter_ids'] = [0]
        c['def_ids'] = [0]
        # c['parent'] = -1
        # c['anchors'] = [np.zeros((3,))]
        c['tree'] = Tree(np.array([0]), root_vertex=0, skip_checks=True)  # Independent part is a single node tree.

        model = dict()
        model['defs'] = [d]
        model['filters'] = [f]
        model['components'] = [c]

        model['maxsize'] = siz[1:]
        model['len'] = 1 + np.prod(siz)
        model['interval'] = 10

        model['feature_pyramid'] = self.feature_pyramid

        # model = self._poswarp(model, pos_) todo: this initialize the weight of the filters to be average feateare
        return Model.model_from_dict(model)

    def _poswarp(self, model, pos):
        # Update independent part model's filter by averaging its hog feature.
        warped = self._warppos(model, pos)
        s = model['filters'][0]['w'].shape  # filter size
        num = len(warped)
        feats = np.empty((np.prod(s), num))
        # for c, im in enumerate(warped):
        #     feat = self.features_pyramid.extract_feature(im)
        #     feats[:, c] = feat.pixels.ravel()
        # w = np.mean(feats, axis=1)
        w = np.empty((np.prod(s),), dtype=np.double)
        scores = np.sum(w * w)
        w2 = w.reshape(s)
        model['filters'][0]['w'] = np.copy(w2)
        model['obj'] = -scores
        return model

    def _warppos(self, model, pos):
        # Load the images, crop and resize them to a predefined shape.
        f = model['components'][0]['filter_ids'][0]  # potentially redundant, f == 0 (but check the rest first.)
        siz = model['filters'][f]['w'].shape[1:]
        spacial_scale = self.feature_pyramid.spacial_scale
        pixels = [spacial_scale * siz[0], spacial_scale * siz[1]]
        num_pos = len(pos)
        heights = np.empty((num_pos,))
        widths = np.empty((num_pos,))
        for i, el in enumerate(pos):
            heights[i] = el['box_y2'][0] - el['box_y1'][0] + 1
            widths[i] = el['box_x2'][0] - el['box_x1'][0] + 1
        crop_size = ((siz[0] + 2) * spacial_scale, (siz[1] + 2) * spacial_scale)
        warped = []
        for i, p in enumerate(pos):
            padx = spacial_scale * widths[i] / pixels[1]
            pady = spacial_scale * heights[i] / pixels[0]

            im = mio.import_image(p['im'])
            im_cr = im.crop([p['box_y1'][0] - pady, p['box_x1'][0] - padx], [p['box_y2'][0] + pady, p['box_x2'][0] + padx])
            im2 = im_cr.resize(crop_size)
            warped.append(im2.copy())
        return warped

    def _train(self, model, pos, negs, iters, c=0.002, wpos=2, wneg=1, maxsize=4, overlap=0.5):  # todo: used to be 0.6
        r"""
        train(improve) the given model by using dual coordinate solver(qp) with the given positive and negative examples

        Parameters
        ----------
        model: `DPM`
            The model which will be improved using the given positive and negative examples
        pos: `list`
            A list of positive examples(dictionary) which contain associated image part and bounding boxes
        negs: `list`
            A list of negative examples(dictionary) which contain associated image part
        iters: `int`
            The maximum iterations that the qp will be executed.
        c: `double`
            The coefficient that indicates the importance of each example
        wpos: `double`
            The weight of positive example's coeeficient (cpos = c * wpos)
        wneg: `double`
            The weight of negative example's coeeficient (cneg = c * wneg)
        maxsize: `int`
            The maximum size of the training data cache (in GB)
        overlap: `double`
            The minimum overlap ratio in latent positive search

        Returns
        -------
        model: `DPM`
            The model improved by the given positive and negative examples.
        """
        # the size of vectorized model
        length = model.sparse_length()
        # Maximum number of model that fitted into the cache
        nmax = round(maxsize*0.25*10**9/length)
        qp = Qp(model, length, nmax, c, wpos)
        for t in range(iters):
            # get positive examples using latent fitting. Found examples are saved to qp.x
            model.delta = self._poslatent(t, model, qp, pos, overlap)
            if model.delta < 0.001:  # terminate if the score doesn't change much
                break
            qp.svfix = range(qp.n)  # fix positive examples as permanent support vectors
            qp.sv[qp.svfix] = True
            qp.prune()
            qp.opt(0.5)
            model = qp.vec2model(model)
            model.interval = 4

            for i, neg in enumerate(negs):
                print('iter:', t, 'neg:', i, '/', np.size(negs))
                im = mio.import_image(neg['im'])
                box, model = DPMFitter.fit_with_bb(im,  model, -1, [], 0, i, -1, qp)
                if np.sum(qp.sv) == nmax:
                    break

            qp.opt()
            model = qp.vec2model(model)
            root_scores = np.sort(qp.score_pos())[::-1]
            # compute minimum score on positive example (with raw, unscaled features)
            model.thresh = root_scores[np.ceil(np.size(root_scores)*0.05)]
            model.interval = 10
            model.lb = qp.lb
            model.ub = qp.ub
        model.feature_pyramid = None  # there is error when trying to save ConvFeaturePyramid
        return model

    def _poslatent(self, t, model, qp, poses, overlap, visualize=False):
        r"""
        Get positive examples using latent fitting. A true bbox is used to check the overlapping with the fitting
        results. The positive examples are saved into qp by calling qp.write

        Parameters
        ----------
        t: `int`
            The number of iteration(s)
        model: `DPM`
            The model which will be improved using the given positive and negative examples
        qp: `Qp`
            The dual coordinate solver which created by the given model
        poses: `list`
            A list of positive examples(dictionary) which contain associated image part and bounding boxes
        overlap: `double`
            The minimum overlap ratio in latent positive search

        Returns
        -------
        delta: `double`
            The different of scores before and after the model is trained(improved)
        """
        num_pos = np.size(poses)
        model.interval = 5
        score0 = qp.score_pos()
        qp.n = 0
        old_w = qp.w
        qp = qp.model2qp(model)
        assert(scipy.linalg.norm(old_w - qp.w) < 10**-5)

        for i, pos in enumerate(poses):
            print('iter:', t, 'pos:', i, '/', num_pos)
            num_parts = np.size(pos['box_x1'])
            bbox = dict()
            bbox['box'] = np.zeros((num_parts, 4))
            bbox['c'] = pos['gmixid'] # to link the index from matlab properly
            for p in range(num_parts):
                bbox['box'][p, :] = [pos['box_x1'][p], pos['box_y1'][p], pos['box_x2'][p], pos['box_y2'][p]]
            im = mio.import_image(pos['im'])
            im, bbox['box'] = self._croppos(im, bbox['box'])
            box, model = DPMFitter.fit_with_bb(im,  model, -np.inf, bbox, overlap, i, 1, qp)
            if visualize and np.size(box) > 0:
                cc, pick = non_max_suppression_fast(clip_boxes([box]), 0.3)
                lns = bb_to_lns([box], pick)
                im.landmarks['tmp'] = PointCloud(lns[0])
                im2 = im.crop_to_landmarks_proportion(0.2, group='tmp')
                im2.view_landmarks(render_numbering=True, group='tmp')
        delta = np.inf
        if t > 0:
            score1 = qp.score_pos()
            loss0 = np.sum(np.maximum(1 - score0, np.zeros_like(score0)))
            loss1 = np.sum(np.maximum(1 - score1, np.zeros_like(score1)))
            assert(loss1 <= loss0)
            if loss0 != 0:
                delta = abs((loss0 - loss1) / loss0)
        assert(qp.n <= qp.x.shape[1])
        return delta

    def _fully_train(self, model, pos, negs, iters, c=0.002, wpos=2, wneg=1, maxsize=4, overlap=0.5):  # todo: used to be 0.6
        r"""
        train(improve) the given model by using dual coordinate solver(qp) with the given positive and negative examples

        Parameters
        ----------
        model: `DPM`
            The model which will be improved using the given positive and negative examples
        pos: `list`
            A list of positive examples(dictionary) which contain associated image part and bounding boxes
        negs: `list`
            A list of negative examples(dictionary) which contain associated image part
        iters: `int`
            The maximum iterations that the qp will be executed.
        c: `double`
            The coefficient that indicates the importance of each example
        wpos: `double`
            The weight of positive example's coeeficient (cpos = c * wpos)
        wneg: `double`
            The weight of negative example's coeeficient (cneg = c * wneg)
        maxsize: `int`
            The maximum size of the training data cache (in GB)
        overlap: `double`
            The minimum overlap ratio in latent positive search

        Returns
        -------
        model: `DPM`
            The model improved by the given positive and negative examples.
        """
        # the size of vectorized model
        example_size = model.sparse_length()
        # Maximum number of model that fitted into the cache
        maximum_examples = round(maxsize*0.25*10**9/example_size)
        qp = Qp(model, example_size, maximum_examples, c, wpos)
        for t in range(iters):
            # get positive examples using latent fitting. Found examples are saved to qp.x
            model.delta = self._feature_poslatent(t, model, qp, pos, overlap)
            if model.delta < 0.001:  # terminate if the score doesn't change much
                break
            qp.svfix = range(qp.n)  # fix positive examples as permanent support vectors
            qp.sv[qp.svfix] = True
            qp.prune()
            qp.opt(iters=5)
            # qp.opt(0.5)
            model = qp.vec2model(model)
            model.interval = 4

            # # todo: if only want the positive alpha, need to fix multiple update
            # alphas, examples, labels = qp.get_support_vector_examples()
            # print(alphas.shape)
            # print(labels.shape)
            # print(examples.shape)
            #
            # examples_feature = model.feature_pyramid.vgg.train_from_qp(examples, labels, alphas, True,
            #                                                            '/vol/atlas/homes/ks3811/pickles/vgg_conv3_3/training_feature/tmp.ckpt')
            # qp.clear_example()
            # qp.update_multiple_exs(examples_feature)

            for i, neg in enumerate(negs):
                print('iter:', t, 'neg:', i, '/', np.size(negs))
                im = mio.import_image(neg['im'])
                padding = (model.maxsize[0]-1-1, model.maxsize[1]-1-1)
                box, model = DPMFitter.fit_for_feature(im,  model, -1, [], 0, i, -1, qp)
                if np.sum(qp.sv) == maximum_examples:
                    break

            qp.opt(iters=5)
            # qp.opt()
            model = qp.vec2model(model)
            root_scores = np.sort(qp.score_pos())[::-1]
            # compute minimum score on positive example (with raw, unscaled features)
            model.thresh = root_scores[np.ceil(np.size(root_scores)*0.05)]
            model.interval = 10
            model.lb = qp.lb
            model.ub = qp.ub

            # print(qp.boxes)
            # print(qp.svfix)
            if t < iters - 1:
                # alphas, examples, labels = qp.get_support_vector_examples_for_regression()
                examples, labels = qp.get_support_vector_examples_for_classify()
                print('examples', examples.shape)
                print('labels', np.sum(labels==68))
                # model.feature_pyramid.vgg.train_from_qp(examples, labels, alphas, False)  #regression
                model.feature_pyramid.vgg.train_from_qp(examples, labels)
                qp.clear_example()

        model.feature_pyramid = None  # there is error when trying to save ConvFeaturePyramid

        return model

    def _feature_poslatent(self, t, model, qp, poses, overlap, visualize=False):
        r"""
        Get positive examples using latent fitting. A true bbox is used to check the overlapping with the fitting
        results. The positive examples are saved into qp by calling qp.write

        Parameters
        ----------
        t: `int`
            The number of iteration(s)
        model: `DPM`
            The model which will be improved using the given positive and negative examples
        qp: `Qp`
            The dual coordinate solver which created by the given model
        poses: `list`
            A list of positive examples(dictionary) which contain associated image part and bounding boxes
        overlap: `double`
            The minimum overlap ratio in latent positive search

        Returns
        -------
        delta: `double`
            The different of scores before and after the model is trained(improved)
        """
        num_pos = np.size(poses)
        model.interval = 5
        score0 = qp.score_pos()
        qp.n = 0
        old_w = qp.w
        qp = qp.model2qp(model)
        assert(scipy.linalg.norm(old_w - qp.w) < 10**-5)

        for i, pos in enumerate(poses):
            print('iter:', t, 'pos:', i, '/', num_pos)
            num_parts = np.size(pos['box_x1'])
            bbox = dict()
            bbox['box'] = np.zeros((num_parts, 4))
            bbox['c'] = pos['gmixid'] # to link the index from matlab properly
            for p in range(num_parts):
                bbox['box'][p, :] = [pos['box_x1'][p], pos['box_y1'][p], pos['box_x2'][p], pos['box_y2'][p]]
            im = mio.import_image(pos['im'])
            im, bbox['box'] = self._croppos(im, bbox['box'])
            # box, model = DPMFitter.fit_with_bb(im,  model, -np.inf, bbox, overlap, i, 1, qp)
            box, model = DPMFitter.fit_for_feature(im,  model, -np.inf, bbox, overlap, i, 1, qp)
            box = box[-1]
            # first_part = box['original_image'][38, :, :, :]
            # part_image = Image(first_part)
            # part_image.view()
            # assert(False)
            qp.boxes.append(box)
            print('boxes size :', len(qp.boxes))
            # print('qp.n', qp.n)
            # print('qp.a', np.sum(qp.a > 0))
            if visualize and np.size(box) > 0:
                im.view()
                cc, pick = non_max_suppression_fast(clip_boxes([box]), 0.3)
                lns = bb_to_lns([box], pick)
                from menpo.shape import PointCloud
                im.landmarks['tmp'] = PointCloud(lns[0])
                im2 = im.crop_to_landmarks_proportion(0.2, group='tmp')
                im2.view_landmarks(render_numbering=True, group='tmp')
        delta = np.inf
        if t > 0:
            score1 = qp.score_pos()
            loss0 = np.sum(np.maximum(1 - score0, np.zeros_like(score0)))
            loss1 = np.sum(np.maximum(1 - score1, np.zeros_like(score1)))
            # assert(loss1 <= loss0)
            if loss0 != 0:
                delta = abs((loss0 - loss1) / loss0)
        assert(qp.n <= qp.x.shape[1])
        return delta

    @staticmethod
    def _croppos(im, box):
        # crop positive example to speed up latent search.
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
        # compute initial deformable coefficients(anchors) by averaging the distance of each positive example's part
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
            points = np.empty((nparts, 2, idx.shape[0]))

            for j, id_j in enumerate(idx):
                magic_number = 20.0  # the 20th percentile of the bbox size of each example.
                maxsize = magic_number/self.feature_pyramid.spacial_scale
                scale0 = boxsize[id_j]/maxsize
                points[:, :, j] = pos[id_j]['pts']/scale0
                # points[:, :, j] = pos[id_j]['pts']/self.feature_pyramid.spacial_scale

            def_tmp = (points[:, :, :] - points[par, :, :])
            def_tmp = np.mean(def_tmp, axis=2)
            defs.append(def_tmp[1:, :])
        return defs

    def build_mixture_model(self, models, defs):
        # Combine the filters learned independently of each part and defs learned from positive examples into a
        # mixture model.
        # Diverging from the original code in the sense that the root filter IS NOT in position 0, but
        # in it's position within the ln points. Additionally, there is a new field called root_id, which
        # shows the precise position of that root filter.
        conf = self.config
        model_ = {}

        assert(len(defs) == len(conf['mixture_poolid']))
        model_['maxsize'] = models[0].maxsize
        model_['len'] = 0
        model_['interval'] = models[0].interval

        # Combine the filters
        model_['filters'] = []
        for m in models:    # for each possible parts
            if np.size(m) > 0:
                f = dict()
                f['w'] = m.filters[0]['w']
                f['i'] = model_['len']
                model_['len'] += np.size(f['w'])
                model_['filters'].append(f)

        # combine the defs
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
        model_['feature_pyramid'] = self.feature_pyramid
        return Model.model_from_dict(model_)

    def train_final_component(self, pickle_dev, index):
        file_name = 'component_further_' + str(index)
        component_file_name, component_time_file_name = self._get_files_names(file_name)
        index = int(index)
        try:
            fp = os.path.join(pickle_dev, component_file_name)
            model = mio.import_pickle(fp)
        except ValueError:
            model = self.train_component(pickle_dev, index)
            model.feature_pyramid = self.feature_pyramid
            start = time.time()
            pos, neg = self._get_pie_image_info(pickle_dev)
            pos = filter(lambda p: p['gmixid'] == index, pos)
            pos = self._ln2box(pos)
            model = self._train(model, pos, neg, 2)
            fp = os.path.join(pickle_dev, component_file_name)
            mio.export_pickle(model, fp)
            stop = time.time()
            fp = os.path.join(pickle_dev, component_time_file_name)
            mio.export_pickle(stop-start, fp)
        return model

    def train_component(self, pickle_dev, index, fully_train=False):
        file_name = 'component_' + str(index)
        component_file_name, component_time_file_name = self._get_files_names(file_name)
        index = int(index)
        try:
            fp = os.path.join(pickle_dev, component_file_name)
            model = mio.import_pickle(fp)
        except ValueError:
            init_model = self.build_model(pickle_dev, index)
            start = time.time()
            pos, neg = self._get_pie_image_info(pickle_dev)
            pos = filter(lambda p: p['gmixid'] == index, pos)
            pos = self._ln2box(pos)
            # pos = [pos[0]]
            k = min(len(neg), 200)
            # k = min(len(neg), 1)
            kneg = neg[0:k]
            if fully_train:
                model = self._fully_train(init_model, pos, kneg, 1)
            else:
                model = self._train(init_model, pos, kneg, 1)
            fp = os.path.join(pickle_dev, component_file_name)
            mio.export_pickle(model, fp)
            stop = time.time()
            fp = os.path.join(pickle_dev, component_time_file_name)
            mio.export_pickle(stop-start, fp)
        return model

    def build_model(self, pickle_dev, index):
        parts_models = self.merge_parts(pickle_dev)

        # Combine the filters learned independently of each part and defs learned from positive examples into a
        # mixture model.
        # Diverging from the original code in the sense that the root filter IS NOT in position 0, but
        # in it's position within the ln points. Additionally, there is a new field called root_id, which
        # shows the precise position of that root filter.
        conf = self.config
        model_ = dict()
        model_['maxsize'] = parts_models[0].maxsize
        model_['len'] = 0
        model_['interval'] = parts_models[0].interval

        # Combine the filters
        model_['filters'] = []
        for part in conf['mixture_poolid'][index]:    # for each possible parts
            f = dict()
            part_model = parts_models[part]
            f['w'] = part_model.filters[0]['w']
            f['i'] = model_['len']
            model_['len'] += np.size(f['w'])
            model_['filters'].append(f)

        # combine the defs
        model_['defs'] = []
        model_['components'] = []
        component = dict()
        def_ids = []
        defs = self.build_defs(pickle_dev, model_['maxsize'], index)
        if np.size(defs) == 0:
            model_['components'].append(component)
        nd = np.size(model_['defs'])
        d = dict()
        d['i'] = model_['len']
        d['w'] = np.array([0])
        d['anchor'] = np.zeros(3,)
        model_['defs'].append(d)
        model_['len'] += np.size(d['w'])
        def_ids.append(nd)
        for j, def_ in enumerate(defs):
            nd = np.size(model_['defs'])
            d = dict()
            d['i'] = model_['len']
            d['w'] = np.array([0.01, 0, 0.01, 0])
            d['anchor'] = np.array([round(def_[0]) + 1, round(def_[1]) + 1, 0])
            model_['defs'].append(d)
            model_['len'] += np.size(d['w'])
            def_ids.append(nd)
        component['def_ids'] = def_ids
        component['filter_ids'] = range(len(conf['mixture_poolid'][index]))
        parents = self._get_parents_lns(index)
        num_parts = np.size(parents)
        pairs = zip(parents, range(num_parts))
        tree_matrix = csr_matrix(([1] * (num_parts-1), (zip(*pairs[1:]))), shape=(num_parts, num_parts))
        component['tree'] = Tree(tree_matrix, root_vertex=0, skip_checks=True)
        model_['components'].append(component)
        model_['feature_pyramid'] = self.feature_pyramid
        return Model.model_from_dict(model_)

    def merge_parts(self, pickle_dev):
        # this method assume that each parts are all trained independently already.
        parts_models = []
        try:
            fp = os.path.join(pickle_dev, 'parts_model_merge.pkl')
            parts_models = mio.import_pickle(fp)
        except ValueError:
            start = time.time()
            for i in xrange(self.config['partpoolsize']):
                file_name = 'parts_model_' + str(i) + '.pkl'
                fp = os.path.join(pickle_dev, file_name)
                print(fp)
                part_model = mio.import_pickle(fp) if os.path.isfile(fp) else None
                parts_models.append(part_model)
            fp = os.path.join(pickle_dev, 'parts_model_merge.pkl')
            mio.export_pickle(parts_models, fp)
            stop = time.time()
            fp = os.path.join(pickle_dev, 'parts_model_merge_time.pkl')
            mio.export_pickle(stop-start, fp)
        return parts_models

    def train_part(self, pickle_dev, index):
        pos, neg = self._get_jpg_pie_image_info(pickle_dev)

        pos = self._ln2box(pos)
        spos = self._split(pos)
        k = min(len(neg), 200)
        kneg = neg[0:k]

        file_name = 'parts_model_' + str(index)
        parts_model_file_name, parts_model_time_file_name = self._get_files_names(file_name)
        index = int(index)
        try:
            fp = os.path.join(pickle_dev, file_name)
            part = mio.import_pickle(fp)
        except ValueError:
            start = time.time()
            init_model = self._init_model(spos[index])
            # spos[index] = spos[index][-1:]
            # kneg = kneg[-1:]
            part = self._train(init_model, spos[index], kneg, iters=4)
            fp = os.path.join(pickle_dev, parts_model_file_name)
            mio.export_pickle(part, fp, overwrite=True)
            stop = time.time()
            fp = os.path.join(pickle_dev, parts_model_time_file_name)
            mio.export_pickle(stop-start, fp)
        return part

    def build_defs(self, pickle_dev, maxsize, index):
        # compute initial deformable coefficients(anchors) by averaging the distance of each positive example's part
        pos, neg = self._get_pie_image_info(pickle_dev)
        pos = self._ln2box(pos)

        conf = self.config
        pos_len = len(pos)
        gmixids = np.empty((pos_len,), dtype=np.int32)
        box_size = np.empty((pos_len,))
        for i, p in enumerate(pos):
            box_size[i] = p['box_x2'][0] - p['box_x1'][0] + 1
            gmixids[i] = p['gmixid'] * 1

        i = int(index)
        parts_num = len(conf['mixture_poolid'][i])
        par = self._get_parents_lns(i)
        idx = np.where(i == gmixids)[0]
        points = np.empty((parts_num, 2, idx.shape[0]))

        for j, id_j in enumerate(idx):
            magic_number = 20.0  # the 20th percentile of the bbox size of each example.
            maxsize = magic_number/self.feature_pyramid.spacial_scale
            scale0 = box_size[id_j]/maxsize
            points[:, :, j] = pos[id_j]['pts']/scale0
            # points[:, :, j] = pos[id_j]['pts']/self.feature_pyramid.spacial_scale

        def_tmp = (points[:, :, :] - points[par, :, :])
        def_tmp = np.mean(def_tmp, axis=2)
        return def_tmp[1:, :]


# this class will be rename to DPM once the current experiment with ~ 700 positive examples finish otherwise pickle
# will complain that class Model is undefined. (the experiment was run when the class is still name Model)
class Model(object):

    def __init__(self, filters=None, defs=None, components=None, interval=10, maxsize=None, len=-1, lb=0,
                 ub=0, delta=0, feature_pyramid=None, thresh=0):
        self.filters = filters
        self.defs = defs
        self.components = components
        self.interval = interval
        self.maxsize = maxsize
        self.len = len
        self.lb = lb
        self.ub = ub
        self.delta = delta
        self.feature_pyramid = feature_pyramid
        self.thresh = thresh

    @classmethod
    def model_from_dict(cls, model):
        # todo: currently DPM model is convert directly from dict. Change part of the code in DPMLearner so that it
        # returns the model directly
        filters = model['filters'] if 'filters' in model else None
        defs = model['defs'] if 'defs' in model else None
        components = model['components'] if 'components' in model else None
        interval = model['interval'] if 'interval' in model else 10
        maxsize = model['maxsize'] if 'maxsize' in model else None
        len = model['len'] if 'len' in model else -1
        lb = model['lb'] if 'lb' in model else 0
        ub = model['ub'] if 'ub' in model else 0
        feature_pyramid = model['feature_pyramid'] if 'feature_pyramid' in model else HogFeaturePyramid()
        thresh = model['thresh'] if 'thresh' in model else 0
        return cls(filters, defs, components, interval, maxsize, len, lb, ub, feature_pyramid=feature_pyramid,
                   thresh=thresh)

    def sparse_length(self):
        # Number of entries needed to encode a block-scarce representation
        length = -1
        for tempcnt, comp in enumerate(self.components):
            if not comp:
                continue
            numblocks = 0
            feats = np.zeros((int(self.len),))
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

            # 1 is used to encode the length itself
            # 2 * blocks are for each block start and end indexes
            # int(np.sum(feats)) is sum of this model component filters and defs
            n = 1 + 2 * numblocks + int(np.sum(feats))
            length = max(length, n)  # length is the maximum length of any model component length
        return length

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


class FeaturePyramid(object):

    def __init__(self, feature_size, spacial_scale):
        self.feature_size = feature_size
        self.spacial_scale = spacial_scale

    def extract_feature(self, img):
        raise NotImplementedError()

    def extract_pyramid(self, img, interval=10, pyramid_pad=(3, 3)):
        spacial_scale = self.spacial_scale
        shift = self.shift
        sc = 2 ** (1. / interval)
        img_size = (img.shape[0], img.shape[1])
        max_scale = int(log_m(min(img_size)*1./(5*spacial_scale))/log_m(sc)) + 1
        feats, scales, shifts = {}, {}, {}
        for i in range(interval):
            sc_l = 2.0 / sc ** i
            scaled = img.rescale(sc_l)
            feats[i] = self.extract_feature(scaled)
            scales[i] = sc_l

            for j in range(i + interval, max_scale + interval, interval):
                scaled = scaled.rescale(0.5)
                feats[j] = self.extract_feature(scaled)
                scales[j] = 0.5 * scales[j-interval]

        expected_size = img.shape[0]
        feats_np = {}  # final feats (keep only numpy array, get rid of the image)
        for k, val in scales.iteritems():
            scales[k] = spacial_scale * 1. / val
            shifts[k] = shift * 1. / val
            # scales[k] = scales[i] = (expected_size * 1.0) / feats[i].shape[1]
            feats_np[k] = np.pad(feats[k], ((0, 0), (int(pyramid_pad[1]), int(pyramid_pad[1])),
                                            (int(pyramid_pad[0]), int(pyramid_pad[0]))), 'constant')
        return feats_np, scales, shifts


    def extract_image_pyramid(self, img, interval=10):
        spacial_scale = self.spacial_scale
        sc = 2 ** (1. / interval)
        img_size = (img.shape[0], img.shape[1])
        max_scale = int(log_m(min(img_size)*1./(5*spacial_scale))/log_m(sc)) + 1
        imgs = {}
        for i in range(interval):
            sc_l = 2.0 / sc ** i
            scaled = img.rescale(sc_l)
            imgs[i] = scaled

            for j in range(i + interval, max_scale + interval, interval):
                scaled = scaled.rescale(0.5)
                imgs[j] = scaled
        return imgs


class HogFeaturePyramid(FeaturePyramid):

    def __init__(self):
        img_size = 40  # this usually need to be perfect scale number
        img = Image(np.zeros((1, img_size, img_size)))
        feature = hog(img, mode='sparse', algorithm='zhuramanan')
        feature_size = feature.pixels.shape[0]
        spacial_scale = 4  # img_size//feature.shape[0]
        super(HogFeaturePyramid, self).__init__(feature_size, spacial_scale)

    def extract_feature(self, img):
        return hog(img, mode='sparse', algorithm='zhuramanan', cell_size=self.spacial_scale)

    def extract_pyramid(self, img, interval=10, pyramid_pad=(3, 3)):
        # Construct the feature pyramid. For the time being, it has similar conventions as in Ramanan's code.
        spacial_scale = self.spacial_scale
        sc = 2 ** (1. / interval)
        img_size = (img.shape[0], img.shape[1])
        max_scale = int(log_m(min(img_size)*1./(5*spacial_scale))/log_m(sc)) + 1
        feats, scales = {}, {}
        for i in range(interval):
            sc_l = 1. / sc ** i
            scaled = img.rescale(sc_l)
            feats[i] = hog(scaled, mode='sparse', algorithm='zhuramanan', cell_size=spacial_scale/2)
            scales[i] = 2 * sc_l
            feats[i + interval] = hog(scaled, mode='sparse', algorithm='zhuramanan', cell_size=spacial_scale)
            scales[i + interval] = sc_l

            for j in range(i + interval, max_scale, interval):
                scaled = scaled.rescale(0.5)
                feats[j + interval] = hog(scaled, mode='sparse', algorithm='zhuramanan', cell_size=spacial_scale)
                scales[j + interval] = 0.5 * scales[j]

        feats_np = {}  # final feats (keep only numpy array, get rid of the image)
        expected_size = img.shape[0]
        for k, val in scales.iteritems():
            scales[k] = spacial_scale * 1. / val
            feats_np[k] = np.pad(feats[k].pixels, ((0, 0), (int(pyramid_pad[1]), int(pyramid_pad[1])),
                                                   (int(pyramid_pad[0]), int(pyramid_pad[0]))), 'constant')
        return feats_np, scales, None


class Qp(object):

    def __init__(self, model, length, nmax, c, wpos):
        # Define global QP problem
        #
        # (Primal) min_{w,e}  .5*||w||^2 + sum_i e_i
        #               s.t.   w*x_j >= b_j - e_i for j in Set_i, for all i
        #
        # (Dual)   max_{a}   -.5*sum_ij a_i*(x_i*x_j)*a_j + sum_i b_i*a_i
        #               s.t.                  a_i >= 0
        #                    sum_(j in Set_i) a_j <= 1
        #
        #   where w = sum_i a_i*x_i
        num_id = 5  # 5 = [label, id, level, posX, poY]
        nmax = int(nmax)
        self.nmax = nmax  # number of maximum examples that can be fitted in memory
        self.x = np.zeros((length, nmax), dtype=np.float32)     # x_i
        self.i = np.zeros((num_id, nmax), dtype=np.int)         # id of each example
        self.b = np.zeros((nmax,), dtype=np.float32)            # b_i
        self.d = np.zeros((nmax,))                              # ||x_i||^2
        self.a = np.zeros((nmax,))                              # a_i
        self.sv = np.zeros((nmax,), dtype=np.bool)              # indicating if x_i is the support vector
        self.w = np.zeros((length,))                            # sum_i a_i*x_i
        self.l = np.array([0], dtype=np.double)                 # sum_i b_i*a_i
        self.n = 0                                              # number of constraints/examples
        self.ub = 0.0                                           # .5*||qp.w||^2 + C*sum_i e_i
        self.lb = 0.0                                           # -.5*sum_ij a_i*(x_i*x_j)*a_j + sum_i b_i*a_i
        self.lb_old = 0.0
        self.svfix = []                                         # pointers to examples that are always kept in memory
        self.boxes = []

        # Put a Gaussian regularization or "prior" on w given by (w0, wreg) where mean = w0, cov^(-1/2) = wreg
        # qp.w = (w - w0)*wreg -> w = qp.w/wreg + w0
        # qp.x = x*c/wreg      -> x = qp.x*wreg/c
        # w*x = (qp.w/wreg + w0)*(qp.x*wreg/c)  (w*x give the score)
        #     = (qp.w + wreg*w0)*qp.x/c
        self.cpos = c * wpos
        self.cneg = c
        self.wreg = []
        self.w0 = []
        self.non_neg = []
        self.model2qp(model)

    def score_pos(self):
        # compute the score of each positive examples
        y = self.i[0, 0:self.n]
        i = np.array(np.nonzero(y)[0], dtype=np.intc)
        w = self.w + self.w0 * self.wreg
        scores = score(w, self.x, i)/self.cpos
        return scores

    def score_neg(self):
        # compute the score of the most recent negative example
        w = - (self.w + self.w0 * self.wreg)
        # -1 bc current n point to the next free space
        scores = score(w, self.x, np.array([self.n - 1],  dtype=np.intc)) / self.cneg
        return scores

    @staticmethod
    def slow_score(w, x):
        # equivalent to score implemented in cpp. used for debugging
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
        # save the example(configurations) returned by fitting into qp's set of examples(x)
        if self.n >= self.nmax:
            return

        label = ex['id'][0] > 0
        c = self.cpos if label else self.cneg
        #  ids = np.sort([block['i'] for block in ex['blocks']])

        bias = 1
        norm = 0
        i = self.n
        j = 0
        self.x[:, i] = 0  # zero out filters and defs of other irrelevant components's parts.
        self.x[j, i] = len(ex['blocks'])

        for block in ex['blocks']:
            n = np.size(block['x'])
            i1 = block['i']
            i2 = i1 + n
            ids = range(i1, i2)
            x = np.array(block['x'])
            x = x if label else -x
            x = x.ravel()  # x.reshape(n, 1)
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

    def write_multiple_exs(self, indexes, filter_index, def_index, filters, defs, will_be_update=False):
        # save the examples(configurations) returned by fitting into qp's set of examples(x)
        # this function consume more memory while significantly reduce the time of writing into qp.x especially
        # in the case of negative examples where a lot of examples are created
        # print(filters.shape)
        # print(defs.shape)

        ex_nums = np.shape(indexes)[1]
        num_parts = np.size(filter_index)

        # if the number of examples are larger than available space, only chose to fit the first ones.
        if self.n + ex_nums > self.nmax:
            ex_nums = np.size(self.a) - self.n
            indexes.resize(indexes.shape[0], ex_nums, refcheck=False)
            filters.resize(filters.shape[0], filters.shape[1], ex_nums, refcheck=False)
            defs.resize(defs.shape[0], defs.shape[1], ex_nums, refcheck=False)

        label = indexes[0][0] > 0
        c = self.cpos if label else self.cneg

        bias = np.ones((ex_nums,), dtype=np.float32)
        norm = np.zeros((ex_nums,), dtype=np.float32)
        i = range(self.n, self.n+ex_nums)
        j = 0
        self.x[:, i] = 0  # zero out filters and defs of other irrelevant components's parts.
        self.x[j, i] = 2*num_parts

        # Special case for the root node where the def is 1 to match with bias. Moved outside the loop for performance
        cv = 0
        x = np.ones((1, 1, ex_nums))
        j, bias, norm = self._create_block(bias, c, cv, def_index, x, i, j, label, norm)
        j, bias, norm = self._create_block(bias, c, cv, filter_index, filters, i, j, label, norm)
        for cv in range(1, num_parts):
            j, bias, norm = self._create_block(bias, c, cv, def_index, defs, i, j, label, norm)
            j, bias, norm = self._create_block(bias, c, cv, filter_index, filters, i, j, label, norm)

        self.d[i] = norm
        self.b[i] = c*bias
        self.i[:, i] = indexes
        self.sv[i] = True
        self.n += ex_nums

        if will_be_update:
            self.indexes, self.filter_index, self.def_index, self.defs = indexes, filter_index, def_index, defs

    def update_multiple_exs(self, filters):
        assert self.n == 0
        self.write_multiple_exs(self.indexes, self.filter_index, self.def_index, filters, self.defs)

    def _create_block(self, bias, c, cv, indexes, values, i, j, label, norm):
        i1 = indexes[cv]
        i2 = i1 + values.shape[1]
        ids = range(i1, i2)
        x = values[cv, :, :]
        x = x if label else -x
        bias = bias - np.sum(self.w0[ids][:, np.newaxis] * x, axis=0)
        x *= c
        x = x / self.wreg[ids][:, np.newaxis]
        self.x[j + 1, i] = i1
        self.x[j + 2, i] = i2
        self.x[j + 3:j + 3 + i2 - i1, i] = x
        norm += np.sum(x * x, axis=0)  # np.dot(np.transpose(x), x)
        j += 3 + i2 - i1 - 1
        return j, bias, norm

    def prune(self):
        # when sv is full, only keep positive examples and the negative examples whose whose alpha(a) is active.
        if np.all(self.sv):
            self.sv = self.a > 0
            print(np.where(self.sv)[0])
            print(self.svfix)
            self.sv[self.svfix] = True

        idxs = np.where(self.sv)[0]
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
        print(idxs)
        print(len(self.boxes))
        if len(self.boxes) >= np.max(idxs) + 1:
            self.boxes = [self.boxes[i] for i in idxs.tolist()]
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
        idxs = np.where(self.a > 0)[0]
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
        # equivalent to lib_comb implemented in cpp. used for debugging
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
        # basic building block for optimising qp for the given set of examples
        idxs = np.where(self.sv)[0]
        np.random.shuffle(idxs)
        idxs = np.array(idxs, dtype=np.intc)
        n = np.size(idxs)
        assert(n > 0)
        #loss = qp_one_sparse(self.x, np.array(self.i, dtype=np.intc), self.b, self.d, self.a, self.w, np.array(self.non_neg, dtype=np.intc), self.sv.view(dtype=np.int8), self.l, 1, idxs)
        loss = self.qp_one_sparse(1, idxs)
        self.refresh()
        self.sv[self.svfix] = True
        self.lb_old = self.lb
        self.lb = self.update_lb()
        self.ub = self.update_ub(loss)
        print('lb : ', self.lb, 'ub : ', self.ub)
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
        # checkout section 2.2 in http://arxiv.org/pdf/1312.1743v2.pdf
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
        # convert qp weight(w) to model
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
        if True:  # todo: change True to debug once tested thoroughly
            self.model2qp(model)
            w_from_updated_model = self.actual_w().astype(np.double)
            if not np.all(np.absolute(original_w - w_from_updated_model) < 10**-5):
                print(np.where(np.absolute(original_w - w_from_updated_model) >= 10**-5))
        return model

    def model2qp(self, model):
        # update qp weight(w) from model
        w = np.zeros((int(model.len),))
        w0 = np.zeros_like(w)
        wreg = np.ones_like(w0)
        non_neg = []
        for x in model.defs:
            l = np.size(x['w'])
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
                l = np.size(x['w'])
                j = np.array(range(x['i'], x['i'] + l))
                w[j] = np.copy(x['w'])
        non_neg = np.array(non_neg, dtype=np.intc).ravel() if np.size(non_neg) > 0 else []
        self.w = (w - w0) * wreg
        self.wreg = wreg
        self.w0 = w0
        self.non_neg = non_neg
        return self

    def obtimise(self, model, iters=1000):
        if self.lb < 0 or self.n == np.size(self.a):
            self.mult(iters=iters)
            self.prune()
        else:
            self.one()
        model = self.vec2model(model)
        # print('qp.n', self.n)
        # print('qp.a', np.sum(self.a > 0))
        return model

    def get_support_vector_examples_for_regression(self):
        # sv_indexes = self.svfix
        sv_indexes = np.where(self.a > 0)[0]
        # sv_indexes = range(self.n)
        alphas = self.a[sv_indexes]
        # print(sv_indexes)
        examples = np.array([np.moveaxis(self.boxes[i]['original_image'], 1, 3) for i in sv_indexes])
        labels = self.i[0, sv_indexes]
        part_examples = list()
        part_alphas = list()
        part_labels = list()
        for i, example in enumerate(examples):
            for part in example:
                part_examples.append(part)
                part_alphas.append(alphas[i])
                part_labels.append(labels[i])
        return np.array(part_alphas), np.array(part_examples), np.array(part_labels)

    def get_support_vector_examples_for_classify(self):
        # sv_indexes = self.svfix
        sv_indexes = np.where(self.a > 0)[0]
        # sv_indexes = range(self.n)
        # print(sv_indexes)
        examples = np.array([np.moveaxis(self.boxes[i]['original_image'], 1, 3) for i in sv_indexes])
        labels = self.i[0, sv_indexes]
        part_examples = list()
        part_alphas = list()
        part_labels = list()
        for i, example in enumerate(examples):
            num_part = example.shape[0]
            print(num_part)
            for j, part in enumerate(example):
                part_examples.append(part)
                part_labels.append(j if labels[i] else num_part)
        return np.array(part_examples), np.array(part_labels)

    def get_support_vector_examples_for_hinge_loss(self):
        # sv_indexes = self.svfix
        sv_indexes = np.where(self.a > 0)[0]
        # sv_indexes = range(self.n)
        # print(sv_indexes)
        examples = np.array([np.moveaxis(self.boxes[i]['original_image'], 1, 3) for i in sv_indexes])
        labels = self.i[0, sv_indexes]
        part_examples = list()
        part_alphas = list()
        part_labels = list()
        for i, example in enumerate(examples):
            num_part = example.shape[0]
            print(num_part)
            for j, part in enumerate(example):
                part_examples.append(part)
                part_labels.append(j if labels[i] else num_part)
        return np.array(part_examples), np.array(part_labels)

    def clear_example(self):
        self.n = 0
        self.a = np.zeros((self.nmax,))
        self.boxes = list()

    def update_example(self, indexes, filter_index, def_index, filters, defs):

        ex_nums = np.shape(indexes)[1]
        num_parts = np.size(filter_index)

        cv = 0
        j, bias, norm = self._create_block(bias, c, cv, def_index, x, i, j, label, norm)
        j, bias, norm = self._create_block(bias, c, cv, filter_index, filters, i, j, label, norm)
        for cv in range(1, num_parts):
            j, bias, norm = self._create_block(bias, c, cv, def_index, defs, i, j, label, norm)
            j, bias, norm = self._create_block(bias, c, cv, filter_index, filters, i, j, label, norm)

        self.d[i] = norm
        self.b[i] = c*bias
        self.i[:, i] = indexes
        self.sv[i] = True
        self.n += ex_nums
