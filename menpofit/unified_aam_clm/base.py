import numpy as np
from scipy.ndimage import gaussian_filter

from menpo.base import name_of_callable
from menpo.model import PCAModel
from menpo.feature import no_op, ndfeature
from menpo.transform import Scale
from menpo.visualize import print_dynamic

from menpofit import checks
from menpofit.builder import (
    build_reference_frame,
    compute_reference_shape,
    rescale_images_to_reference_shape,
    compute_features,
    scale_images,
    warp_images,
)
from menpofit.aam.algorithm.lk import LucasKanadeStandardInterface
from menpofit.clm import CorrelationFilterExpertEnsemble
from menpofit.clm.expert.ensemble import ConvolutionBasedExpertEnsemble
from menpofit.modelinstance import OrthoPDM
from menpofit.transform import DifferentiablePiecewiseAffine, OrthoMDTransform


@ndfeature
def fsmooth(pixels, sigma, mode="constant"):
    return gaussian_filter(pixels, sigma, mode=mode)


class UnifiedAAMCLM(object):
    r"""
    Class for training a multi-scale unified holistic AAM and CLM as
    presented in [1].
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
        The reference shape that will be used for building the AAM. The purpose
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
    expert_ensemble_cls : `subclass` of :map:`ExpertEnsemble`, optional
        The class to be used for training the ensemble of experts. The most
        common choice is :map:`CorrelationFilterExpertEnsemble`.
    patch_shape : (`int`, `int`) or `list` of (`int`, `int`), optional
        The shape of the patches to be extracted. If a `list` is provided,
        then it defines a patch shape per scale.
    context_shape : (`int`, `int`) or `list` of (`int`, `int`), optional
        The context shape for the convolution. If a `list` is provided,
        then it defines a context shape per scale.
    sample_offsets : ``(n_offsets, n_dims)`` `ndarray` or ``None``, optional
        The sample_offsets to sample from within a patch. So ``(0, 0)`` is the
        centre of the patch (no offset) and ``(1, 0)`` would be sampling the
        patch from 1 pixel up the first axis away from the centre. If ``None``,
        then no sample_offsets are applied.
    transform : `subclass` of :map:`DL` and :map:`DX`, optional
        A differential warp transform object, e.g.
        :map:`DifferentiablePiecewiseAffine` or
        :map:`DifferentiableThinPlateSplines`.
    shape_model_cls : `subclass` of :map:`OrthoPDM`, optional
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
    sigma : `float` or ``None``, optional
        If not ``None``, the input images are smoothed with an isotropic
        Gaussian filter with the specified standard deviation.
    boundary : `int`, optional
        The number of pixels to be left as a safe margin on the boundaries
        of the reference frame (has potential effects on the gradient
        computation).
    response_covariance : `int`, optional
        The covariance of the generated Gaussian response.
    patch_normalisation : `callable`, optional
        The normalisation function to be applied on the extracted patches.
    cosine_mask : `bool`, optional
        If ``True``, then a cosine mask (Hanning function) will be applied on
        the extracted patches.
    verbose : `bool`, optional
        If ``True``, then the progress of building the model will be printed.

    References
    ----------
    .. [1] J. Alabort-i-Medina, and S. Zafeiriou. "Unifying holistic and
        parts-based deformable model fitting", Proceedings of the IEEE
        Conference on Computer Vision and Pattern Recognition (CVPR), 2015.
    """

    def __init__(
        self,
        images,
        group=None,
        holistic_features=no_op,
        reference_shape=None,
        diagonal=None,
        scales=(0.5, 1.0),
        expert_ensemble_cls=CorrelationFilterExpertEnsemble,
        patch_shape=(17, 17),
        context_shape=(34, 34),
        sample_offsets=None,
        transform=DifferentiablePiecewiseAffine,
        shape_model_cls=OrthoPDM,
        max_shape_components=None,
        max_appearance_components=None,
        sigma=None,
        boundary=3,
        response_covariance=2,
        patch_normalisation=no_op,
        cosine_mask=True,
        verbose=False,
    ):
        # Check parameters
        checks.check_diagonal(diagonal)
        scales = checks.check_scales(scales)
        n_scales = len(scales)
        holistic_features = checks.check_callable(holistic_features, n_scales)
        shape_model_cls = checks.check_callable(shape_model_cls, n_scales)
        max_shape_components = checks.check_max_components(
            max_shape_components, n_scales, "max_shape_components"
        )
        max_appearance_components = checks.check_max_components(
            max_appearance_components, n_scales, "max_appearance_components"
        )
        # Assign attributes
        self.expert_ensemble_cls = checks.check_callable(expert_ensemble_cls, n_scales)
        self.expert_ensembles = []
        self.patch_shape = checks.check_patch_shape(patch_shape, n_scales)
        self.context_shape = checks.check_patch_shape(context_shape, n_scales)
        self.holistic_features = holistic_features
        self.transform = transform
        self.diagonal = diagonal
        self.scales = scales
        self.max_shape_components = max_shape_components
        self.max_appearance_components = max_appearance_components
        self.reference_shape = reference_shape
        self.shape_model_cls = shape_model_cls
        self.sigma = sigma
        self.boundary = boundary
        self.sample_offsets = sample_offsets
        self.response_covariance = response_covariance
        self.patch_normalisation = patch_normalisation
        self.cosine_mask = cosine_mask
        self.shape_models = []
        self.appearance_models = []
        self.expert_ensembles = []

        self._train(images=images, group=group, verbose=verbose)

    def _build_reference_frame(self, mean_shape):
        return build_reference_frame(mean_shape, boundary=self.boundary)

    def _warp_images(
        self, images, shapes, reference_shape, scale_index, prefix, verbose
    ):
        reference_frame = build_reference_frame(reference_shape)
        return warp_images(
            images,
            shapes,
            reference_frame,
            self.transform,
            prefix=prefix,
            verbose=verbose,
        )

    def _train(self, images, group=None, verbose=False):
        checks.check_landmark_trilist(images[0], self.transform, group=group)
        self.reference_shape = compute_reference_shape(
            [i.landmarks[group] for i in images], self.diagonal, verbose=verbose
        )

        # normalize images
        images = rescale_images_to_reference_shape(
            images, group, self.reference_shape, verbose=verbose
        )
        if self.sigma:
            images = [fsmooth(i, self.sigma) for i in images]

        # Build models at each scale
        if verbose:
            print_dynamic("- Building models\n")

        feature_images = []
        # for each scale (low --> high)
        for j in range(self.n_scales):
            if verbose:
                if len(self.scales) > 1:
                    scale_prefix = "  - Scale {}: ".format(j)
                else:
                    scale_prefix = "  - "
            else:
                scale_prefix = None

            # Handle holistic features
            if j == 0 and self.holistic_features[j] == no_op:
                # Saves a lot of memory
                feature_images = images
            elif (
                j == 0 or self.holistic_features[j] is not self.holistic_features[j - 1]
            ):
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_images = compute_features(
                    images,
                    self.holistic_features[j],
                    prefix=scale_prefix,
                    verbose=verbose,
                )
            # handle scales
            if self.scales[j] != 1:
                # Scale feature images only if scale is different than 1
                scaled_images = scale_images(
                    feature_images, self.scales[j], prefix=scale_prefix, verbose=verbose
                )
            else:
                scaled_images = feature_images

            # Extract potentially rescaled shapes
            scale_shapes = [i.landmarks[group] for i in scaled_images]

            # Build the shape model
            if verbose:
                print_dynamic("{}Building shape model".format(scale_prefix))

            shape_model = self._build_shape_model(scale_shapes, j)
            self.shape_models.append(shape_model)

            # Obtain warped images - we use a scaled version of the
            # reference shape, computed here. This is because the mean
            # moves when we are incrementing, and we need a consistent
            # reference frame.
            scaled_reference_shape = Scale(self.scales[j], n_dims=2).apply(
                self.reference_shape
            )
            warped_images = self._warp_images(
                scaled_images,
                scale_shapes,
                scaled_reference_shape,
                j,
                scale_prefix,
                verbose,
            )

            # obtain appearance model
            if verbose:
                print_dynamic("{}Building appearance model".format(scale_prefix))

            appearance_model = PCAModel(warped_images)
            # trim appearance model if required
            if self.max_appearance_components[j] is not None:
                appearance_model.trim_components(self.max_appearance_components[j])
            # add appearance model to the list
            self.appearance_models.append(appearance_model)

            expert_ensemble = self.expert_ensemble_cls[j](
                images=scaled_images,
                shapes=scale_shapes,
                patch_shape=self.patch_shape[j],
                patch_normalisation=self.patch_normalisation,
                cosine_mask=self.cosine_mask,
                context_shape=self.context_shape[j],
                sample_offsets=self.sample_offsets,
                prefix=scale_prefix,
                verbose=verbose,
            )
            self.expert_ensembles.append(expert_ensemble)

            if verbose:
                print_dynamic("{}Done\n".format(scale_prefix))

    def _build_shape_model(self, shapes, scale_index):
        return self.shape_model_cls[scale_index](
            shapes, max_n_components=self.max_shape_components[scale_index]
        )

    @property
    def n_scales(self):
        """
        Returns the number of scales.

        :type: `int`
        """
        return len(self.scales)

    @property
    def _str_title(self):
        return "Unified Holistic AAM and CLM"

    def shape_instance(self, shape_weights=None, scale_index=-1):
        r"""
        Generates a novel shape instance given a set of shape weights. If no
        weights are provided, the mean shape is returned.

        Parameters
        ----------
        shape_weights : ``(n_weights,)`` `ndarray` or `list` or ``None``, optional
            The weights of the shape model that will be used to create a novel
            shape instance. If ``None``, the weights are assumed to be zero,
            thus the mean shape is used.
        scale_index : `int`, optional
            The scale to be used.

        Returns
        -------
        instance : `menpo.shape.PointCloud`
            The shape instance.
        """
        if shape_weights is None:
            shape_weights = [0]
        sm = self.shape_models[scale_index].model
        return sm.instance(shape_weights, normalized_weights=True)

    def instance(self, shape_weights=None, appearance_weights=None, scale_index=-1):
        r"""
        Generates a novel instance of the AAM part of the model given a set of
        shape and appearance weights. If no weights are provided, then the mean
        AAM instance is returned.

        Parameters
        ----------
        shape_weights : ``(n_weights,)`` `ndarray` or `list` or ``None``, optional
            The weights of the shape model that will be used to create a novel
            shape instance. If ``None``, the weights are assumed to be zero,
            thus the mean shape is used.
        appearance_weights : ``(n_weights,)`` `ndarray` or `list` or ``None``, optional
            The weights of the appearance model that will be used to create a
            novel appearance instance. If ``None``, the weights are assumed
            to be zero, thus the mean appearance is used.
        scale_index : `int`, optional
            The scale to be used.

        Returns
        -------
        image : `menpo.image.Image`
            The AAM instance.
        """
        if shape_weights is None:
            shape_weights = [0]
        if appearance_weights is None:
            appearance_weights = [0]
        sm = self.shape_models[scale_index].model
        am = self.appearance_models[scale_index]

        shape_instance = sm.instance(shape_weights, normalized_weights=True)
        appearance_instance = am.instance(appearance_weights, normalized_weights=True)

        return self._instance(scale_index, shape_instance, appearance_instance)

    def random_instance(self, scale_index=-1):
        r"""
        Generates a random instance of the AAM part of the model.

        Parameters
        ----------
        scale_index : `int`, optional
            The scale to be used.

        Returns
        -------
        image : `menpo.image.Image`
            The AAM instance.
        """
        sm = self.shape_models[scale_index].model
        am = self.appearance_models[scale_index]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = np.random.randn(sm.n_active_components)
        shape_instance = sm.instance(shape_weights, normalized_weights=True)
        appearance_weights = np.random.randn(sm.n_active_components)
        appearance_instance = am.instance(appearance_weights, normalized_weights=True)

        return self._instance(scale_index, shape_instance, appearance_instance)

    def _instance(self, scale_index, shape_instance, appearance_instance):
        template = self.appearance_models[scale_index].mean()
        landmarks = template.landmarks["source"]

        reference_frame = build_reference_frame(shape_instance)

        transform = self.transform(reference_frame.landmarks["source"], landmarks)

        return appearance_instance.as_unmasked(copy=False).warp_to_mask(
            reference_frame.mask, transform, warp_landmarks=True
        )

    def build_fitter_interfaces(self, sampling):
        r"""
        Method that builds the correct fitting interface for a
        :map:`UnifiedAAMCLMFitter`.

        Parameters
        ----------
        sampling : `list` of `int` or `ndarray` or ``None``
            It defines a sampling mask per scale. If `int`, then it
            defines the sub-sampling step of the sampling mask. If `ndarray`,
            then it explicitly defines the sampling mask. If ``None``, then no
            sub-sampling is applied.

        Returns
        -------
        fitter_interfaces : `list`
            The `list` of fitting interfaces per scale.
        """
        interfaces = []
        for am, sm, s in zip(self.appearance_models, self.shape_models, sampling):
            template = am.mean()
            md_transform = OrthoMDTransform(
                sm, self.transform, source=template.landmarks["source"]
            )

            interface = LucasKanadeStandardInterface(
                am, md_transform, template, sampling=s
            )
            interfaces.append(interface)
        return interfaces

    def appearance_reconstructions(self, appearance_parameters, n_iters_per_scale):
        r"""
        Method that generates the appearance reconstructions given a set of
        appearance parameters. This is to be combined with a
        :map:`UnifiedAAMCLMResult` object, in order to generate the appearance
        reconstructions of a fitting procedure.

        Parameters
        ----------
        appearance_parameters : `list` of ``(n_params,)`` `ndarray`
            A set of appearance parameters per fitting iteration. It can be
            retrieved as a property of an :map:`UnifiedAAMCLMResult` object.
        n_iters_per_scale : `list` of `int`
            The number of iterations per scale. This is necessary in order to
            figure out which appearance parameters correspond to the model of
            each scale. It can be retrieved as a property of a
            :map:`UnifiedAAMCLMResult` object.

        Returns
        -------
        appearance_reconstructions : `list` of `menpo.image.Image`
            `List` of the appearance reconstructions that correspond to the
            provided parameters.
        """
        appearance_reconstructions = []
        previous = 0
        for scale, n_iters in enumerate(n_iters_per_scale):
            for c in appearance_parameters[previous : previous + n_iters + 1]:
                instance = self.appearance_models[scale].instance(c)
                appearance_reconstructions.append(instance)
            previous = n_iters + 1
        return appearance_reconstructions

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
     - Ensemble of experts class: {}
       - {} experts
       - {} class
       - Patch shape: {} x {}
       - Patch normalisation: {}
       - Context shape: {} x {}
       - Cosine mask: {}
     - Appearance model class: {}
       - {} appearance components
     - Shape model class: {}
       - {} shape components
       - {} similarity transform parameters"""
        for k, s in enumerate(self.scales):
            scales_info.append(
                lvl_str_tmplt.format(
                    s,
                    name_of_callable(self.holistic_features[k]),
                    name_of_callable(self.expert_ensemble_cls[k]),
                    self.expert_ensembles[k].n_experts,
                    name_of_callable(self.expert_ensembles[k]._icf),
                    self.expert_ensembles[k].patch_shape[0],
                    self.expert_ensembles[k].patch_shape[1],
                    name_of_callable(self.expert_ensembles[k].patch_normalisation),
                    self.expert_ensembles[k].context_shape[0],
                    self.expert_ensembles[k].context_shape[1],
                    self.expert_ensembles[k].cosine_mask,
                    name_of_callable(self.appearance_models[k]),
                    self.appearance_models[k].n_components,
                    name_of_callable(self.shape_models[k]),
                    self.shape_models[k].model.n_components,
                    self.shape_models[k].n_global_parameters,
                )
            )

        scales_info = "\n".join(scales_info)

        if self.transform is not None:
            transform_str = "Images warped with {} transform".format(
                name_of_callable(self.transform)
            )
        else:
            transform_str = "No image warping performed"

        cls_str = r"""{class_title}
  - Images scaled to diagonal: {diagonal:.2f}
  - {transform}
  - Scales: {scales}
{scales_info}
    """.format(
            class_title=self._str_title,
            diagonal=diagonal,
            transform=transform_str,
            scales=self.scales,
            scales_info=scales_info,
        )
        return cls_str
