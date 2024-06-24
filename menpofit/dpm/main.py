from __future__ import print_function

import os
import sys
import menpo.io as mio
import numpy as np
import scipy.io
import time
from menpo.shape import Tree, bounding_box, PointCloud
from menpofit.dpm import DPMLearner, DPMFitter, Model, non_max_suppression_fast, clip_boxes, bb_to_lns_eval, \
    HogFeaturePyramid, bb_to_lns
from scipy.sparse import csr_matrix
from menpofit.error import euclidean_error
from menpofit.error.base import bb_normalised_error
from menpofit.visualize.base import plot_cumulative_error_distribution


def debugging():

    # Uncomment this to start the learner which currently use images and configurations base on original matlab code
    pickle_dev = '/vol/atlas/homes/ks3811/pickles/refactor2'
    dpm_learner = DPMLearner()
    dpm_learner.train_part(pickle_dev, 0)
    #
    # mat_model = get_matlab_model()

    # Uncomment these to use the model learned from python
    pickle_dev = '/vol/atlas/homes/ks3811/pickles/refactor'
    model_name = 'final2.pkl'
    model = get_model(pickle_dev, model_name, HogFeaturePyramid())

    pickle_dev = '/vol/atlas/homes/ks3811/pickles/conv_scale_2'
    model_name = 'component_6.pkl'
    alex_scale_2_model = get_model(pickle_dev, model_name, HogFeaturePyramid())

    im = mio.import_builtin_asset('takeo.ppm')
    # im = mio.import_builtin_asset.lenna_png()
    # im = mio.import_image('/vol/hci2/Databases/video/MultiPIE/session02/png/038/01/19_0/038_02_01_190_05.png')

    pickle_dev = '/vol/atlas/homes/ks3811/pickles/condor/first/*'
    models = get_models(pickle_dev, HogFeaturePyramid())

    visualise_fit_image(im, models)


def visualise_fit_image(im, models, threshold=0, index=0):
    start = time.time()
    boxes = list()
    if isinstance(models, list):
        boxes = DPMFitter.fast_fit_from_models(im, models, threshold, return_once=True)
    elif isinstance(models, Model):
        boxes = DPMFitter.fast_fit_from_model(im, models, threshold, return_once=True)
    print('Found {0} configuration(s).'.format(np.size(boxes)))
    boxes.sort(key=lambda item: item['s'], reverse=True)
    cc, pick = non_max_suppression_fast(clip_boxes(boxes), 0.3)
    lns = bb_to_lns(boxes, pick)
    end = time.time()
    print('Fitting time taken: {0} seconds'.format(end - start))
    im.landmarks['TEST'] = PointCloud(lns[index])
    crop_im = im.crop_to_landmarks_proportion(0.2, group='TEST')
    crop_im.view_landmarks(render_numbering=True, group='TEST')
    return lns


def get_models(pickle, feature_pyramid):
    models = list(mio.import_pickles(pickle))
    for model in models:
        mixture = int(model.path.stem.split('_')[-1].split('.')[0])
        model.mixture = mixture
        if feature_pyramid is not None:
            model.feature_pyramid = feature_pyramid
    return models


def get_model(pickle, model_name, feature_pyramid):
    file_name = os.path.join(pickle, model_name)
    model = mio.import_pickle(file_name)
    model.interval = 10  # Use deeper pyramid when detecting actual objects
    if feature_pyramid is not None:
        model.feature_pyramid = feature_pyramid
    return model


def get_matlab_model():
    matlab_folder = '/vol/atlas/homes/grigoris/external/dpm_ramanan/face-release1.0-basic/'
    matlab_file_name = os.path.join(matlab_folder, 'face_p146_small.mat')
    matlab_model = scipy.io.loadmat(matlab_file_name)

    # Convert matlab model into python model format
    model = matlab_model['model']
    maxsize = model['maxsize'][0][0][0]
    sbin = model['sbin'][0][0][0][0]
    interval = model['interval'][0][0][0][0]
    thresh = min(0, model['thresh'][0][0][0][0])
    components, anchors = get_components(model)
    filters_all = get_filters(model)
    defs_all = get_defs(model, anchors)

    model = dict()
    model['maxsize'] = maxsize
    model['sbin'] = sbin
    model['interval'] = interval
    model['filters'] = filters_all
    model['defs'] = defs_all
    model['components'] = components
    model['thresh'] = thresh
    model['interval'] = 10  # Use deeper pyramid when detecting actual objects
    model['feature_pyramid'] = HogFeaturePyramid()
    return Model.model_from_dict(model)


def get_components(model):
    # Component contains information about filters and defs indexes as well as each component tree structure
    components = []
    comp_m = model['components'][0][0][0] # *_m = * in matlab
    anchors = []
    for c in range(len(comp_m)):    # 13 components
        cm = comp_m[c][0]
        filter_ids = []
        def_ids = []
        parents = []
        # filter_index = []  # indexes are not need for fitting. but just leave it right now
        # def_index = [] 
        num_parts = len(cm)
        for k in range(num_parts):
            filter_id = cm[k]['filterid'][0][0]
            filter_ids.append(filter_id - 1)  # -1 due to matlab numbering
            def_id = cm[k]['defid'][0][0]  # -1 due to matlab numbering
            def_ids.append(def_id - 1)
            parents.append(cm[k]['parent'][0][0] - 1)
            # def_index.append(model['defs'][0][0][0][def_id - 1][1][0][0])  # -1 due to matlab numbering
            # filter_index.append(model['filters'][0][0][0][filter_id - 1][1][0][0])  # -1 due to matlab numbering
            x = model['defs'][0][0][0][def_id - 1]  # -1 due to matlab numbering
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
        tt = np.rollaxis(ff[i]['w'], 2)  # rollaxis since matlab format is (5, 5, 32) while menpo hog is (31, 5, 5)
        tt = tt[:-1, :, :]  # ignore matlab hog last index (32 in matlab vs 31 in menpo).
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


def eval_multipie_mixture_models_with_hog():
    feature_pyramid = HogFeaturePyramid()
    eval_mixture_models('/vol/atlas/homes/ks3811/pickles/eval',
                        '/vol/atlas/homes/ks3811/pickles/condor/first/*', feature_pyramid)


def eval_mixture_models(test_pickle, model_pickle, feature_pyramid=None, return_once=True):
    _, _, tests = _get_pie_image_info(test_pickle)
    models = get_models(model_pickle, feature_pyramid)
    results = list()
    for test in tests:
        img = mio.import_image(test['im'])
        boxes = DPMFitter.fast_fit_from_models(img, models, 0, return_once)
        result = dict()
        result['im'] = test['im']
        result['boxes'] = boxes
        results.append(result)
        print(test['im'], result)
    fp = os.path.join(test_pickle, model_pickle.replace('/', '_') + '_results_.pkl')
    mio.export_pickle(results, fp)


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
        test = _c['test']
        neg = _c['neg']
    except ValueError:  # Data in pickle does not exist
        start = time.time()
        train_list = [50, 50, 50, 50, 50, 50, 300, 50, 50, 50, 50, 50, 50]
        test_list = [50, 50, 50, 50, 50, 50, 300, 50, 50, 50, 50, 50, 50]
        pos = []
        test = []
        print('Collecting info for the positive images.')
        for gmixid, poses in enumerate(pos_data):
            create_test = False
            pos_count = 0
            test_count = 0
            for img in poses['images']:
                if pos_count >= train_list[gmixid]:
                    create_test = True
                if test_count >= test_list[gmixid]:
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

                aux = dict()
                aux['pts'] = scipy.io.loadmat(anno_file_name)['pts'][DPMLearner._get_anno2tree(gmixid), :]
                aux['im'] = file_name
                aux['gmixid'] = gmixid
                if create_test:
                    test_count += 1
                    test.append(aux)
                else:
                    pos_count += 1
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
        _c['test'] = test
        mio.export_pickle(_c, fp)
        stop = time.time()
        fp = os.path.join(pickle_dev, 'data_time.pkl')
        mio.export_pickle(stop-start, fp)
    return pos, neg, test


def compare_multipie_cumulative_errors():
    hog_results = mio.import_pickle('/homes/ks3811/Phd/1_year/_vol_atlas_homes_ks3811_pickles_condor_first_all_results_.pkl')
    hog = 'HOG'

    results1 = mio.import_pickle('/vol/atlas/homes/ks3811/pickles/eval/_vol_atlas_homes_ks3811_pickles_mdm_first_*_results_.pkl')
    results2 = mio.import_pickle('/homes/ks3811/Phd/1_year/_vol_atlas_homes_ks3811_pickles_mdm_first_*0-300_results_.pkl')
    results3 = mio.import_pickle('/homes/ks3811/Phd/1_year/_vol_atlas_homes_ks3811_pickles_mdm_first_*400-700_results_.pkl')
    mdm_results = results1 + results2 + results3
    mdm = 'MDM'

    results1 = mio.import_pickle('/homes/ks3811/Phd/1_year/_vol_atlas_homes_ks3811_pickles_conv_scale_4_first_*300-400_results_.pkl')
    results2 = mio.import_pickle('/homes/ks3811/Phd/1_year/_vol_atlas_homes_ks3811_pickles_conv_scale_4_first_*0-300_results_.pkl')
    alex_results = results1 + results2
    alex = 'ALEX'

    list_of_results = [hog_results, mdm_results, alex_results]
    alg_names = list([hog, mdm, alex])

    test_pickle = '/vol/atlas/homes/ks3811/pickles/eval'
    compare_cumulative_object_detection_errors(test_pickle, list_of_results, alg_names)
    compare_cumulative_pose_estimation_errors(test_pickle, list_of_results, alg_names)
    compare_cumulative_landmark_localisation_errors(test_pickle, list_of_results, alg_names)


def compare_cumulative_object_detection_errors(test_pickle, list_of_results, alg_names, error_thresh=0.5):
    compare_cumulative_errors(test_pickle, list_of_results, alg_names, condition_function=_face_detection_condition,
                              error_function=_face_detection_error, error_thresh=error_thresh,
                              error_range=[0., 1., 0.1], x_label='Intersection-over-union errors')


def _face_detection_condition(result, _):
    return 'pts' not in result


def _face_detection_error(result, ground_truth):
    return  1 - bounding_box_overlap_ratio(PointCloud(result['pts']), PointCloud(ground_truth['pts']))


def compare_cumulative_pose_estimation_errors(test_pickle, list_of_results, alg_names, error_thresh=15):
    compare_cumulative_errors(test_pickle, list_of_results, alg_names, condition_function=_pose_estimation_condition,
                              error_function=_pose_estimation_error, error_thresh=error_thresh, error_range=[0, 45, 15],
                              x_label='Pose estimation error (in degree)')


def _pose_estimation_condition(result, _):
    return 'mixture' not in result


def _pose_estimation_error(result, ground_truth):
    correct_indexes = [0, 1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9]
    return  np.abs(ground_truth['gmixid'] - correct_indexes[result['mixture']])*15


def compare_cumulative_landmark_localisation_errors(test_pickle, list_of_results, alg_names, error_thresh=0.05):
    compare_cumulative_errors(test_pickle, list_of_results, alg_names,
                              condition_function=_landmark_localisation_condition,
                              error_function=_landmark_localisation_error, error_thresh=error_thresh, error_range=None,
                              x_label=None)


def _landmark_localisation_condition(result, ground_truth):
    return ('pts' not in result) or (np.size(result['pts']) != np.size(ground_truth['pts']))


def _landmark_localisation_error(result, ground_truth):
    return bb_normalised_error(euclidean_error, result['pts'], ground_truth['pts'], norm_type='avg_edge_length',
                               norm_shape=ground_truth['pts'])


def _51_landmark_localisation_error(result, ground_truth):
    excluded_indexes = range(51, 59 + 1) + range(60, 67 + 1)
    included_indexes = [index for index in range(68) if index not in excluded_indexes]
    return bb_normalised_error(euclidean_error, result['pts'][included_indexes], ground_truth['pts'][included_indexes],
                               norm_type='avg_edge_length', norm_shape=ground_truth['pts'][included_indexes])


def compare_cumulative_errors(test_pickle, list_of_results, alg_names, condition_function, error_function,
                              error_thresh=0.0, error_range=None, x_label=None):
    _, _, tests = _get_pie_image_info(test_pickle)
    im_names = [x['im'] for x in tests]
    cumulative_errors = list()
    legend_entries = list()
    for i, results in enumerate(list_of_results):
        errors = list()
        for result in results:
            test = tests[np.where(np.array(im_names) == result['im'])[0]]
            if condition_function(result, test):
                error = np.inf
                errors.append(error)
                continue
            error = error_function(result, test)
            errors.append(error)
        ratio_at_thresh = round((np.size(np.where(np.array(errors) <= error_thresh)) + 0.0)/np.size(results)*100, 2)
        legend_entries.append(alg_names[i] + ' (' + str(ratio_at_thresh) + '%)')
        cumulative_errors.append(errors)
    plot_cumulative_error_distribution(cumulative_errors, legend_entries=legend_entries, error_range=error_range,
                                       x_label=x_label)


def compare_matlab_python_cumulative_landmark_localisation_errors(thresh_error=0.05):
    python_errors = mio.import_pickle('/vol/atlas/homes/ks3811/pickles/test/m_multipie_d_multipie_errors.pkl')
    python = round(float(np.size(np.where(np.array(python_errors) <= thresh_error)))/np.size(python_errors) * 100, 2)

    f = open('/vol/atlas/homes/ks3811/pickles/eval/matlab', 'r')
    matlab_errors = list()
    for line in f:
        matlab_errors.append(float(line))
    matlab = round(float(np.size(np.where(np.array(matlab_errors) <= thresh_error)))/np.size(matlab_errors) * 100, 2)

    errors = [matlab_errors, python_errors]
    plot_cumulative_error_distribution(errors, legend_entries=['Matlab (' + str(matlab) + '%)',
                                                               'Python (' + str(python) + '%)'])


def overlap(bbox_a, bbox_b):
    (a_top, a_left), (a_bottom, a_right) = bbox_a.bounds()
    (b_top, b_left), (b_bottom, b_right) = bbox_b.bounds()
    h_overlaps = (a_left <= b_right) and (a_right >= b_left)
    v_overlaps = (a_bottom >= b_top) and (a_top <= b_bottom)
    return h_overlaps and v_overlaps


def intersect(bbox_a, bbox_b):
    if overlap(bbox_a, bbox_b):
        (a_top, a_left), (a_bottom, a_right) = bbox_a.bounds()
        (b_top, b_left), (b_bottom, b_right) = bbox_b.bounds()
        o_top = max(a_top, b_top)
        o_bottom = min(a_bottom, b_bottom)
        o_left = max(a_left, b_left)
        o_right = min(a_right, b_right)
        return bounding_box((o_top, o_left), (o_bottom, o_right))
    else:  # Empty box
        return bounding_box((0., 0.), (0., 0.))


def bounding_box_overlap_ratio(bbox_a, bbox_b, ratio_type='union'):
    intersect_ab = intersect(bbox_a, bbox_b)
    area_intersect_ab = np.prod(intersect_ab.range())
    area_a = np.prod(bbox_a.range())
    area_b = np.prod(bbox_b.range())
    if ratio_type == 'union':  # divide by union of bbox_a and bbox_b
        return area_intersect_ab / (area_a + area_b - area_intersect_ab)
    elif ratio_type == 'min':  # divide by minimum of bbox_a and bbox_b
        return area_intersect_ab / min(area_a, area_b)
    else:
        raise ValueError('Unexpected ratio_type: {}'.format(ratio_type))


def detection_error(pc_a, pc_b, ratio_type='union'):
    return bounding_box_overlap_ratio(pc_a, pc_b, ratio_type) >= 0.5

if __name__ == "__main__":
    # sys.exit(compute_errors())
    # sys.exit(eval_multipie_mixture_models_with_hog())
    # sys.exit(compare_results())
    sys.exit(debugging())
    # compare_multipie_cumulative_errors()
