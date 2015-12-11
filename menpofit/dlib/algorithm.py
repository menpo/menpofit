from __future__ import division
import dlib

from .conversion import (copy_dlib_options, pointcloud_to_dlib_rect,
                         bounding_box_pointcloud_to_dlib_fo_detection,
                         dlib_full_object_detection_to_pointcloud,
                         image_to_dlib_pixels)
from menpo.visualize import print_dynamic

from menpofit.result import NonParametricAlgorithmResult


# TODO: document me!
class DlibAlgorithm(object):
    r"""
    """

    def __init__(self, dlib_options, n_iterations=10):
        self.dlib_model = None
        self._n_iterations = n_iterations
        self.dlib_options = copy_dlib_options(dlib_options)
        # T from Kazemi paper - Total number of cascades
        self.dlib_options.cascade_depth = self.n_iterations

    @property
    def n_iterations(self):
        return self._n_iterations

    @n_iterations.setter
    def n_iterations(self, v):
        self._n_iterations = v
        # T from Kazemi paper - Total number of cascades
        self.dlib_options.cascade_depth = self._n_iterations

    def train(self, images, gt_shapes, bounding_boxes, prefix='',
              verbose=False):

        if verbose and self.dlib_options.oversampling_amount > 1:
            n_menpofit_peturbations = len(bounding_boxes[0])
            n_dlib_perturbations = self.dlib_options.oversampling_amount
            total_perturbations = (n_menpofit_peturbations *
                                   n_dlib_perturbations)
            print_dynamic('{}WARNING: Dlib oversampling is being used. '
                          '{} = {} * {} total perturbations will be generated '
                          'by Dlib!\n'.format(prefix, total_perturbations,
                                              n_menpofit_peturbations,
                                              n_dlib_perturbations))

        im_pixels = [image_to_dlib_pixels(im) for im in images]

        detections = []
        for bboxes, im, gt_s in zip(bounding_boxes, images, gt_shapes):
            fo_dets = [bounding_box_pointcloud_to_dlib_fo_detection(bb, gt_s)
                       for bb in bboxes]
            detections.append(fo_dets)

        if verbose:
            print_dynamic('{}Performing Dlib training - please see stdout '
                          'for verbose output provided by Dlib!'.format(prefix))

        # Perform DLIB training
        self.dlib_options.be_verbose = verbose
        self.dlib_model = dlib.train_shape_predictor(
            im_pixels, detections, self.dlib_options)

        for bboxes, pix, fo_dets in zip(bounding_boxes, im_pixels, detections):
            for bb, fo_det in zip(bboxes, fo_dets):
                # Perform prediction
                pred = dlib_full_object_detection_to_pointcloud(
                    self.dlib_model(pix, fo_det.rect))
                # Update bounding box in place
                bb._from_vector_inplace(pred.bounding_box().as_vector())

        if verbose:
            print_dynamic('{}Training Dlib done.\n'.format(prefix))

        return bounding_boxes

    def run(self, image, bounding_box, gt_shape=None, **kwargs):
        # Perform prediction
        pix = image_to_dlib_pixels(image)
        rect = pointcloud_to_dlib_rect(bounding_box)
        pred = dlib_full_object_detection_to_pointcloud(
            self.dlib_model(pix, rect))

        return NonParametricAlgorithmResult(image, [pred], gt_shape=gt_shape)
