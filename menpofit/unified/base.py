import abc
import numpy as np
from copy import deepcopy
from serializablecallable import SerializableCallable
from menpo.model import PCAModel
from menpo.image import Image
from menpo.feature import no_op
from menpo.transform import Scale, Translation, GeneralizedProcrustesAnalysis
from menpo.visualize import print_dynamic, progress_bar_str
from menpo.shape import mean_pointcloud
from menpo.transform import AlignmentUniformScale
from menpofit import checks
from menpofit.transform.piecewiseaffine import DifferentiablePiecewiseAffine
from menpofit.transform import OrthoMDTransform
from menpofit.modelinstance import OrthoPDM
from menpofit.builder import build_reference_frame
from menpofit.clm import CorrelationFilterExpertEnsemble
from scipy.stats import multivariate_normal
from scipy.ndimage import gaussian_filter

fsmooth = lambda x, sigma: gaussian_filter(x, sigma, mode='constant')

def rescale_to_reference_shape(image, reference_shape, group=None, label=None, round='ceil', order=1):
    pc = image.landmarks[group][label]
    scale = AlignmentUniformScale(pc, reference_shape).as_vector().copy()
    return image.rescale(scale, round=round, order=order)

class UnifiedAAMCLM(object):
    r"""
    Class for training a multi-scale unified holistic AAM and CLM as 
    per [1].
    Please see the references for AAMs and CLMs in their respective
    base classes.

    Parameters
    ----------
    images : `list` of `menpo.image.Image`
        The `list` of training images.
    group : `str` or ``None``, optional
        The landmark group that will be used to train the model. If ``None`` and
        the images only have a single landmark group, then that is the one
        that will be used. Note that all the training images need to have the
        specified landmark group.
    holistic_features : `closure` or `list` of `closure`, optional
        The features that will be extracted from the training images. Note
        that the features are extracted before warping the images to the
        reference shape. If `list`, then it must define a feature function per
        scale. Please refer to `menpo.feature` for a list of potential features.
    reference_shape : `menpo.shape.PointCloud` or ``None``, optional
        The reference shape that will be used for building the model. The purpose
        of the reference shape is to normalise the size of the training images.
        The normalization is performed by rescaling all the training images
        so that the scale of their ground truth shapes matches the scale of
        the reference shape. Note that the reference shape is rescaled with
        respect to the `diagonal` before performing the normalisation. If
        ``None``, then the mean shape will be used.
    diagonal : `int` or ``None``, optional
        This parameter is used to rescale the reference shape so that the
        diagonal of its bounding box matches the provided value. In other
        words, this parameter controls the size of the model at the highest
        scale. If ``None``, then the reference shape does not get rescaled.
    scales : `float` or `tuple` of `float`, optional
        The scale value of each scale. They must provided in ascending order,
        i.e. from lowest to highest scale. If `float`, then a single scale is
        assumed.
    transform : `subclass` of :map:`DL` and :map:`DX`, optional
        A differential warp transform object, e.g.
        :map:`DifferentiablePiecewiseAffine` or
        :map:`DifferentiableThinPlateSplines`.
    shape_model_cls : `subclass` of :map:`PDM`, optional
        The class to be used for building the shape model. The most common
        choice is :map:`OrthoPDM`.
    max_shape_components : `int`, `float`, `list` of those or ``None``, optional
        The number of shape components to keep. If `int`, then it sets the exact
        number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should
        define a value per scale. If a single number, then this will be
        applied to all scales. If ``None``, then all the components are kept.
        Note that the unused components will be permanently trimmed.
    max_appearance_components : `int`, `float`, `list` of those or ``None``, optional
        The number of appearance components to keep. If `int`, then it sets the
        exact number of components. If `float`, then it defines the variance
        percentage that will be kept. If `list`, then it should define a value
        per scale. If a single number, then this will be applied to all
        scales. If ``None``, then all the components are kept. Note that the
        unused components will be permanently trimmed.
    verbose : `bool`, optional
        If ``True``, then the progress of building the model will be printed.

    References
    ----------
    .. [1] J. Alabort-i-Medina, and S. Zafeiriou. "A Unified Framework for
        Compositional Fitting of Active Appearance Models", arXiv:1601.00199.
    """
    def __init__(self, images, expert_ensemble_cls=CorrelationFilterExpertEnsemble, 
                 parts_shape=(17, 17), context_shape=(34, 34), offsets=None, group=None, holistic_features=no_op,
                 reference_shape=None, diagonal=None, scales=(0.5, 1.0),
                 transform=DifferentiablePiecewiseAffine,
                 shape_model_cls=OrthoPDM, max_shape_components=None,
                 max_appearance_components=None, scale_shapes=False, scale_features=True, sigma=None, 
                 boundary=3, normalize_parts=False, covariance=2, patch_normalisation=no_op, cosine_mask=True, verbose=False):
        # Check parameters
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        n_scales = len(scales)
        holistic_features = checks.check_callable(holistic_features, n_scales)
        shape_model_cls = checks.check_callable(shape_model_cls, n_scales)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_scales, 'max_shape_components')
        max_appearance_components = checks.check_max_components(
            max_appearance_components, n_scales, 'max_appearance_components')
        # Assign attributes
        self.expert_ensemble_cls = checks.check_callable(expert_ensemble_cls,n_scales)
        self.expert_ensembles = []
        self.parts_shape = checks.check_patch_shape(parts_shape, n_scales)
        self.context_shape = checks.check_patch_shape(context_shape, n_scales)
        self.holistic_features = holistic_features
        self.transform = transform
        self.diagonal = diagonal
        self.scales = scales
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.reference_shape = reference_shape
        self._shape_model_cls = shape_model_cls
        self.shape_models = []
        self.appearance_models = []
        self.scale_shapes = scale_shapes
        self.sigma = sigma
        self.boundary = boundary
        self.offsets = offsets
        self.normalize_parts = normalize_parts
        self.covariance = covariance
        self.scale_features = scale_features
        self.patch_normalisation = patch_normalisation
        self.cosine_mask = cosine_mask

        self._train(images=images, group=group, verbose=verbose)

    def _build_reference_frame(self, mean_shape):
        return build_reference_frame(mean_shape, boundary=self.boundary)

    def _warp_images(self, images, shapes, ref_shape, level_str, verbose):
        # compute transforms
        ref_frame = self._build_reference_frame(ref_shape)
        # warp images to reference frame
        warped_images = []
        for c, (i, s) in enumerate(zip(images, shapes)):
            if verbose:
                print_dynamic('{}Warping images - {}'.format(
                    level_str,
                    progress_bar_str(float(c + 1) / len(images),
                                     show_bar=False)))
            # compute transforms
            t = self.transform(ref_frame.landmarks['source'].lms, s)
            # warp images
            warped_i = i.warp_to_mask(ref_frame.mask, t)
            # attach reference frame landmarks to images
            warped_i.landmarks['source'] = ref_frame.landmarks['source']
            warped_images.append(warped_i)
        return warped_images
  
    def _train(self, images, group=None, label=None, verbose=False):
        # compute reference shape
        if self.reference_shape is None:
            self.reference_shape = self._compute_reference_shape(images, group, label, verbose)
        
        # normalize images
        images = self._normalize_images(images, group, label, self.reference_shape,
                                        verbose)
        # build models at each scale
        if verbose:
            print_dynamic('- Building models\n')
        self.shape_models = []
        self.appearance_models = []
        self.expert_ensembles = []
        # for each pyramid level (high --> low)
        for j, s in enumerate(self.scales):
            if verbose:
                if len(self.scales) > 1:
                    level_str = '  - Scale {}: '.format(j)
                else:
                    level_str = '  - '
            else:
                level_str = None

            # obtain image representation
            if j == 0:
                # compute features at highest level
                feature_images = self._compute_features(images, j, level_str,
                                                        verbose)
                level_images = feature_images
            elif self.scale_features:
                # scale features at other levels
                level_images = self._scale_images(feature_images, s,
                                                  level_str, verbose)
            else:
                # scale images and compute features at other levels
                scaled_images = self._scale_images(images, s, level_str,
                                                   verbose)
                level_images = self._compute_features(scaled_images, j,
                                                      level_str, verbose)

            # extract potentially rescaled shapes ath highest level
            level_shapes = [i.landmarks[group][label]
                            for i in level_images]

            # obtain shape representation
            if j == 0 or self.scale_shapes:
                # obtain shape model
                if verbose:
                    print_dynamic('{}Building shape model'.format(level_str))
                shape_model = self._build_shape_model(
                    level_shapes, self.max_shape_components[j])
                # add shape model to the list
                self.shape_models.append(shape_model)
            else:
                # copy precious shape model and add it to the list
                self.shape_models.append(deepcopy(shape_model))

            # obtain warped images
            warped_images = self._warp_images(level_images, level_shapes,
                                              shape_model.mean(), level_str,
                                              verbose)

            # obtain appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(level_str))
            appearance_model = PCAModel(warped_images)
            # trim appearance model if required
            if self.max_appearance_components[j] is not None:
                appearance_model.trim_components(
                    self.max_appearance_components[j])
            # add appearance model to the list
            self.appearance_models.append(appearance_model)

            expert_ensemble = self.expert_ensemble_cls[j](
                    images=level_images, shapes=level_shapes,
                    patch_shape=self.parts_shape[j],
                    patch_normalisation=self.patch_normalisation,
                    cosine_mask=self.cosine_mask,
                    context_shape=self.context_shape[j],
                    sample_offsets=self.offsets,
                    prefix=level_str, verbose=verbose)
            
            self.expert_ensembles.append(expert_ensemble)

            if verbose:
                print_dynamic('{}Done\n'.format(level_str))

        # reverse the list of shape and appearance models so that they are
        # ordered from lower to higher resolution
        self.shape_models.reverse()
        self.appearance_models.reverse()
        self.expert_ensembles.reverse()
        self.scales.reverse()

    def _compute_reference_shape(self, images, group, label, verbose):
        # the reference_shape is the mean shape of the images' landmarks
        if verbose:
            print_dynamic('- Computing reference shape')
        shapes = [i.landmarks[group][label] for i in images]
        ref_shape = mean_pointcloud(shapes)
        # fix the reference_shape's diagonal length if specified
        if self.diagonal:
            x, y = ref_shape.range()
            scale = self.diagonal / np.sqrt(x**2 + y**2)
            ref_shape = Scale(scale, ref_shape.n_dims).apply(ref_shape)
        return ref_shape

    def _normalize_images(self, images, group, label, ref_shape, verbose):
        # normalize the scaling of all images wrt the reference_shape size
        norm_images = []
        for c, i in enumerate(images):
            if verbose:
                print_dynamic('- Normalizing images size: {}'.format(
                    progress_bar_str((c + 1.) / len(images), show_bar=False)))
            i = rescale_to_reference_shape(i, ref_shape, group=group,
                                             label=label)
            if self.sigma:
                i.pixels = fsmooth(i.pixels, self.sigma)
            norm_images.append(i)
        return norm_images

    def _compute_features(self, images, scale_index, level_str, verbose):
        feature_images = []
        for c, i in enumerate(images):
            if verbose:
                print_dynamic(
                    '{}Computing feature space: {}'.format(
                        level_str, progress_bar_str((c + 1.) / len(images),
                                                    show_bar=False)))
            if self.holistic_features:
                i = self.holistic_features[scale_index](i)
            feature_images.append(i)

        return feature_images

    @classmethod
    def _scale_images(cls, images, s, level_str, verbose):
        scaled_images = []
        for c, i in enumerate(images):
            if verbose:
                print_dynamic(
                    '{}Scaling features: {}'.format(
                        level_str, progress_bar_str((c + 1.) / len(images),
                                                    show_bar=False)))
            scaled_images.append(i.rescale(s))
        return scaled_images

    @classmethod
    def _build_shape_model(cls, shapes, max_components):
        r"""
        Builds a shape model given a set of shapes.

        Parameters
        ----------
        shapes: list of :map:`PointCloud`
            The set of shapes from which to build the model.
        max_components: None or int or float
            Specifies the number of components of the trained shape model.
            If int, it specifies the exact number of components to be retained.
            If float, it specifies the percentage of variance to be retained.
            If None, all the available components are kept (100% of variance).

        Returns
        -------
        shape_model: :class:`menpo.model.pca`
            The PCA shape model.
        """

        # centralize shapes
        centered_shapes = [Translation(-s.centre()).apply(s) for s in shapes]
        # align centralized shape using Procrustes Analysis
        gpa = GeneralizedProcrustesAnalysis(centered_shapes)
        aligned_shapes = [s.aligned_source() for s in gpa.transforms]
        # build shape model
        shape_model = PCAModel(aligned_shapes)
        if max_components is not None:
            # trim shape model if required
            shape_model.trim_components(max_components)

        return shape_model

    @property
    def n_scales(self):
        """
        Returns the number of scales.

        :type: `int`
        """
        return len(self.scales)

    @property
    def _str_title(self):
        return 'Unified Holistic AAM and CLM'

    def __str__(self):
        if self.diagonal is not None:
            diagonal = self.diagonal
        else:
            y, x = self.reference_shape.range()
            diagonal = np.sqrt(x ** 2 + y ** 2)

        # Compute scale info strings
        scales_info = []
        lvl_str_tmplt = r"""   - Scale {}
         - Holistic feature: {}
         - Appearance model class: {}
           - {} appearance components
         - Shape model class: {}
           - {} shape components
           - {} similarity transform parameters"""
        for k, s in enumerate(self.scales):
            scales_info.append(lvl_str_tmplt.format(
                s, name_of_callable(self.holistic_features[k]),
                name_of_callable(self.appearance_models[k]),
                self.appearance_models[k].n_components,
                name_of_callable(self.shape_models[k]),
                self.shape_models[k].model.n_components,
                self.shape_models[k].n_global_parameters))

        scales_info = '\n'.join(scales_info)

        if self.transform is not None:
            transform_str = 'Images warped with {} transform'.format(
                name_of_callable(self.transform))
        else:
            transform_str = 'No image warping performed'

        cls_str = r"""{class_title}
     - Images scaled to diagonal: {diagonal:.2f}
     - {transform}
     - Scales: {scales}
    {scales_info}
    """.format(class_title=self._str_title,
               transform=transform_str,
               diagonal=diagonal,
               scales=self.scales,
               scales_info=scales_info)
        return cls_str


