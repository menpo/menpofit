from __future__ import division
from copy import deepcopy
import warnings
import numpy as np
from scipy.stats import multivariate_normal

from menpo.feature import no_op
from menpo.visualize import print_dynamic, print_progress
from menpo.model import PCAInstanceModel, GMRFInstanceModel
from menpo.shape import (DirectedGraph, UndirectedGraph, Tree, PointTree,
                         PointDirectedGraph, PointUndirectedGraph)

from menpofit import checks
from menpofit.base import batch
from menpofit.builder import (compute_features, scale_images, align_shapes,
                              rescale_images_to_reference_shape,
                              extract_patches, MenpoFitBuilderWarning,
                              compute_reference_shape)

class GenerativeAPS(object):
    r"""
    Class for building a Generative Active Pictorial Structures Model.

    Parameters
    ----------
    images : `list` or `generator` of :map:`Image` objects.
        The batch of training images.
    group : `str` or ``None``, optional
        The :map:`LandmarkGroup` of the images to be used for training.
    appearance_graph : `list` of graphs or a single graph or ``None``, optional
        The graph to be used for the appearance :map:`GMRFInstanceModel`
        training. It must be an :map:`UndirectedGraph`. If ``None``, then ...
    shape_graph : `list` of graphs or a single graph or ``None``, optional
        The graph to be used for the shape :map:`GMRFInstanceModel` training.
        It must be an :map:`UndirectedGraph`. If ``None``, then the shape
        model is built using :map:`PCAInstanceModel`.
    deformation_graph : `list` of graphs or a single graph or ``None``, optional
        The graph to be used for the deformation :map:`GMRFInstanceModel`
        training. It must be either a :map:`DirectedGraph` or a :map:`Tree`.
        If ``None``, then the minimum spanning tree of the data is computed.
    reference_shape : :map:`PointCloud` or ``None``, optional
        The reference shape that will be used for normalizing the size of the
        images. If ``None``, then the mean pointcloud will be used.
    holistic_features : `list` of `callable` or a single `callable`, optional
        If `list`, then it must have length equal to the number of scales. If a
        single feature `callable`, then this is the one applied to all scales.
    patch_normalisation : `list` of `callable` or a single `callable`, optional
        If `list`, then it must have length equal to the number of scales. If a
        single patch normalization `callable`, then this is the one applied to
        all scales.
    diagonal : `int` or ``None``, optional
        If not ``None``, then the reference shape is rescaled so that the
        diagonal of its bounding box matches the provided value. If ``None``,
        then no rescaling is performed.
    scales : `tuple` of `float`, optional
        The scales of the pyramid. The `float` scales are defined with
        respect to the initial scale of the images and must be provided from
        low level to high level.
    patch_shape : `list` of `tuple` or a single `tuple`, optional
        If `list`, then it must have length equal to the number of scales. If a
        single patch shape `tuple`, then this is the one applied to all
        scales. The `tuple` must have two `int` numbers corresponding to the
        height and width of the patch, i.e. ``(int, int)``.
    use_procrustes : `bool`, optional
        If ``True``, then Generalized Procrustes Alignment is applied before
        building the shape and deformation models.
    precision_dtype : {``'single', 'double'``}, optional
        The numerical precision of the appearance GMRF precision matrix. Even
        though the precision matrix is stored as a `scipy.sparse` matrix,
        this parameter has a big impact on the amount of memory required by
        the model.
    max_shape_components : `list` of `int` or `int` or ``None``, optional
        The maximum number of shape components to be kept in the shape
        :map:`PCAInstanceModel`. If `list`, then it must have length equal to
        the number of scales. If a single `int`, then this is the one applied
        to all scales. If ``None``, then all the components are kept.
    n_appearance_components : `list` of `int` or `int` or ``None``, optional
        The number of appearance components used for building the appearance
        :map:`GMRFInstanceModel`. If `list`, then it must have length equal to
        the number of scales. If a single `int`, then this is the one applied
        to all scales. If ``None``, the covariance matrix of each edge is
        inverted using `np.linalg.inv`. If `int`, it is inverted using
        truncated SVD using the specified number of components.
    can_be_incremented : `bool`, optional
        In case you intend to incrementally update the model in the future,
        then this flag must be set to ``True`` from the first place. Note
        that if ``True``, the appearance and deformation
        :map:`GMRFInstanceModel` models will occupy double memory.
    batch_size : `int` or ``None``, optional
        If `int`, then `images` may be a `generator`, else, if ``None``,
        we assume that `images` is a `list`.
    verbose : `bool`, optional
        If ``True``, information related to the training progress will be
        printed.

    References
    ----------
    .. [1] E. Antonakos, J. Alabort-i-Medina, and S. Zafeiriou, "Active
        Pictorial Structures", Proceedings of the IEEE Conference on Computer
        Vision and Pattern Recognition (CVPR), Boston, MA, USA, pp. 1872-1882,
        June 2015.
    """
    def __init__(self, images, group=None, appearance_graph=None,
                 shape_graph=None, deformation_graph=None, reference_shape=None,
                 holistic_features=no_op, patch_normalisation=no_op,
                 diagonal=None, scales=(0.5, 1.0), patch_shape=(17, 17),
                 use_procrustes=True, precision_dtype='single',
                 max_shape_components=None, n_appearance_components=None,
                 can_be_incremented=False, batch_size=None, verbose=False):
        # Check arguments
        self.diagonal = checks.check_diagonal(diagonal)
        self.scales = checks.check_scales(scales)
        n_scales = len(self.scales)
        self.patch_shape = checks.check_patch_shape(patch_shape, n_scales)
        self.precision_dtype = checks.check_precision(precision_dtype)
        self.holistic_features = checks.check_features(
            holistic_features, n_scales, 'holistic_features')
        self.patch_normalisation = patch_normalisation
        self.max_shape_components = checks.check_max_components(
            max_shape_components, n_scales, 'max_shape_components')
        self.n_appearance_components = checks.check_max_components(
            n_appearance_components, n_scales, 'n_appearance_components')
        self.is_incremental = can_be_incremented
        self.reference_shape = reference_shape
        self.use_procrustes = use_procrustes

        # Check provided graphs
        self.appearance_graph = checks.check_graph(
            appearance_graph, UndirectedGraph, 'appearance_graph', n_scales)
        self.shape_graph = checks.check_graph(shape_graph, UndirectedGraph,
                                              'shape_graph', n_scales)
        self.deformation_graph = checks.check_graph(
            deformation_graph, [DirectedGraph, Tree], 'deformation_graph',
            n_scales)

        # Initialize models' lists
        self.shape_models = []
        self.appearance_models = []
        self.deformation_models = []

        # Train APS
        self._train(images, increment=False, group=group,
                    batch_size=batch_size, verbose=verbose)

    def _train(self, images, increment=False, group=None, batch_size=None,
               verbose=False):
        # If batch_size is not None, then we may have a generator, else we
        # assume we have a list.
        if batch_size is not None:
            # Create a generator of fixed sized batches. Will still work even
            # on an infinite list.
            image_batches = batch(images, batch_size)
        else:
            image_batches = [list(images)]

        for k, image_batch in enumerate(image_batches):
            if k == 0:
                if self.reference_shape is None:
                    # If no reference shape was given, use the mean of the first
                    # batch
                    if batch_size is not None:
                        warnings.warn('No reference shape was provided. The '
                                      'mean of the first batch will be the '
                                      'reference shape. If the batch mean is '
                                      'not representative of the true mean, '
                                      'this may cause issues.',
                                      MenpoFitBuilderWarning)
                    self.reference_shape = compute_reference_shape(
                        [i.landmarks[group].lms for i in image_batch],
                        self.diagonal, verbose=verbose)

            # After the first batch, we are incrementing the model
            if k > 0:
                increment = True

            if verbose:
                print('Computing batch {}'.format(k))

            # Train each batch
            self._train_batch(
                image_batch, increment=increment, group=group, verbose=verbose)

    def _train_batch(self, image_batch, increment=False, group=None,
                     verbose=False):
        # Rescale to existing reference shape
        image_batch = rescale_images_to_reference_shape(
            image_batch, group, self.reference_shape, verbose=verbose)

        # If the deformation graph was not provided (None given), then compute
        # the MST
        if None in self.deformation_graph:
            graph_shapes = [i.landmarks[group].lms for i in image_batch]
            deformation_mst = _compute_minimum_spanning_tree(
                graph_shapes, root_vertex=0, prefix='- ', verbose=verbose)
            self.deformation_graph = [deformation_mst if g is None else g
                                      for g in self.deformation_graph]

        # Build models at each scale
        if verbose:
            print_dynamic('- Building models\n')

        feature_images = []
        # for each scale (low --> high)
        for j in range(self.n_scales):
            if verbose:
                if len(self.scales) > 1:
                    scale_prefix = '  - Scale {}: '.format(j)
                else:
                    scale_prefix = '  - '
            else:
                scale_prefix = None

            # Handle holistic features
            if j == 0 and self.holistic_features[j] == no_op:
                # Saves a lot of memory
                feature_images = image_batch
            elif (j == 0 or self.holistic_features[j] is not
                  self.holistic_features[j - 1]):
                # Compute features only if this is the first pass through
                # the loop or the features at this scale are different from
                # the features at the previous scale
                feature_images = compute_features(image_batch,
                                                  self.holistic_features[j],
                                                  prefix=scale_prefix,
                                                  verbose=verbose)
            # handle scales
            if self.scales[j] != 1:
                # Scale feature images only if scale is different than 1
                scaled_images = scale_images(feature_images, self.scales[j],
                                             prefix=scale_prefix,
                                             verbose=verbose)
            else:
                scaled_images = feature_images

            # Extract potentially rescaled shapes
            scale_shapes = [i.landmarks[group].lms for i in scaled_images]

            # Apply procrustes to align the shapes if asked
            if self.use_procrustes:
                aligned_shapes = align_shapes(scale_shapes)
            else:
                aligned_shapes = scale_shapes

            # Build the shape model
            if verbose:
                print_dynamic('{}Building shape model'.format(scale_prefix))
            if not increment:
                if j == 0:
                    shape_model = self._build_shape_model(
                        aligned_shapes, self.shape_graph[j], verbose=verbose)
                    self.shape_models.append(shape_model)
                else:
                    self.shape_models.append(deepcopy(shape_model))
            else:
                self.shape_models[j].increment(aligned_shapes, verbose=verbose)

            # Build the deformation model
            if verbose:
                print_dynamic('{}Building deformation model'.format(
                    scale_prefix))
            if not increment:
                if j == 0:
                    deformation_model = self._build_deformation_model(
                        aligned_shapes, self.deformation_graph[j],
                        verbose=verbose)
                    self.deformation_models.append(deformation_model)
                else:
                    self.deformation_models.append(deepcopy(deformation_model))
            else:
                self.deformation_models[j].increment(aligned_shapes,
                                                     verbose=verbose)

            # Obtain warped images
            warped_images = self._warp_images(scaled_images, scale_shapes,
                                              j, scale_prefix, verbose)

            # Build the appearance model
            if verbose:
                print_dynamic('{}Building appearance model'.format(
                    scale_prefix))
            if not increment:
                self.appearance_models.append(self._build_appearance_model(
                    warped_images, self.appearance_graph[j],
                    self.n_appearance_components[j], verbose=verbose))
            else:
                self._increment_appearance_model(
                    warped_images, self.appearance_graph[j],
                    self.appearance_models[j], verbose=verbose)

            if verbose:
                print_dynamic('{}Done\n'.format(scale_prefix))

        # Because we just copy the shape model, we need to wait to trim
        # it after building each model. This ensures we can have a different
        # number of components per level
        for j, sm in enumerate(self.shape_models):
            max_sc = self.max_shape_components[j]
            if max_sc is not None:
                sm.trim_components(max_sc)

    def increment(self, images, group=None, batch_size=None, verbose=False):
        r"""
        Method that incrementally updates the model with a new batch of
        training images.

        Parameters
        ----------
        images : `list` or `generator` of :map:`Image` objects.
            The batch of training images.
        group : `str` or ``None``, optional
            The :map:`LandmarkGroup` of the images to be used for training.
        batch_size : `int` or ``None``, optional
            If `int`, then `images` may be a `generator`, else, if ``None``,
            we assume that `images` is a `list`.
        verbose : `bool`, optional
            If ``True``, information related to the training progress will be
            printed.
        """
        return self._train(images, increment=True, group=group,
                           verbose=verbose, batch_size=batch_size)

    def _build_shape_model(self, shapes, shape_graph, verbose=False):
        # if the provided graph is None, then apply PCA, else use the GMRF
        if shape_graph is not None:
            return GMRFInstanceModel(
                shapes, shape_graph, mode='concatenation', n_components=None,
                single_precision=False, sparse=False,
                incremental=self.is_incremental,
                verbose=verbose).principal_components_analysis(
                apply_on_precision=False)
        else:
            return PCAInstanceModel(shapes)

    def _build_deformation_model(self, shapes, deformation_graph,
                                 verbose=False):
        return GMRFInstanceModel(
            shapes, deformation_graph, mode='subtraction', n_components=None,
            single_precision=False, sparse=False,
            incremental=self.is_incremental, verbose=verbose)

    def _build_appearance_model(self, images, appearance_graph,
                                n_appearance_components, verbose=False):
        if appearance_graph is not None:
            return GMRFInstanceModel(
                images, appearance_graph, mode='concatenation',
                n_components=n_appearance_components,
                single_precision=self.precision_dtype, sparse=True,
                incremental=self.is_incremental, verbose=verbose)
        else:
            raise NotImplementedError('The full appearance model is not '
                                      'implemented yet.')

    def _increment_appearance_model(self, images, appearance_graph,
                                    appearance_model, verbose=False):
        if appearance_graph is not None:
            appearance_model.increment(images, verbose=verbose)
        else:
            raise NotImplementedError('The full appearance model is not '
                                      'implemented yet.')

    def _warp_images(self, images, shapes, scale_index, prefix, verbose):
        return extract_patches(images, shapes, self.patch_shape[scale_index],
                               normalise_function=self.patch_normalisation,
                               prefix=prefix, verbose=verbose)

    @property
    def n_scales(self):
        """
        The number of scales of the Generative APS.

        :type: `int`
        """
        return len(self.scales)

    @property
    def _str_title(self):
        r"""
        Returns a string containing name of the model.

        :type: `str`
        """
        return 'Generative Active Pictorial Structures'

    def instance(self, shape_weights=None, scale_index=-1, as_graph=False):
        r"""
        Generates an instance of the shape model.

        Parameters
        ----------
        shape_weights : `list` or ``None``, optional
            The weights to be applied on the shape model. If ``None``,
            then the weights are all equal to zero.
        scale_index : `int`, optional
            Defines the scale level, the shape model of which will be used.
        as_graph : `bool`, optional
            If ``True``, then the instance will be returned as a
            :map:`PointTree` or a :map:`PointDirectedGraph`, depending on the
            type of the deformation graph.
        """
        sm = self.shape_models[scale_index]

        # TODO: this bit of logic should to be transferred down to PCAModel
        if shape_weights is None:
            shape_weights = [0]
        n_shape_weights = len(shape_weights)
        shape_weights *= sm.eigenvalues[:n_shape_weights] ** 0.5
        shape_instance = sm.instance(shape_weights)

        if as_graph:
            if isinstance(self.deformation_graph[scale_index], Tree):
                shape_instance = PointTree(
                    shape_instance.points,
                    self.deformation_graph[scale_index].adjacency_matrix,
                    self.deformation_graph[scale_index].root_vertex)
            else:
                shape_instance = PointDirectedGraph(
                    shape_instance.points,
                    self.deformation_graph[scale_index].adjacency_matrix)
        return shape_instance

    def random_instance(self, scale_index=-1, as_graph=False):
        r"""
        Generates an instance of the shape model with random weights.

        Parameters
        ----------
        scale_index : `int`, optional
            Defines the scale level, the shape model of which will be used.
        as_graph : `bool`, optional
            If ``True``, then the instance will be returned as a
            :map:`PointTree` or a :map:`PointDirectedGraph`, depending on the
            type of the deformation graph.
        """
        sm = self.shape_models[scale_index]

        # TODO: this bit of logic should to be transferred down to PCAModel
        shape_weights = (np.random.randn(sm.n_active_components) *
                         sm.eigenvalues[:sm.n_active_components]**0.5)
        shape_instance = sm.instance(shape_weights)

        if as_graph:
            if isinstance(self.deformation_graph[scale_index], Tree):
                shape_instance = PointTree(
                    shape_instance.points,
                    self.deformation_graph[scale_index].adjacency_matrix,
                    self.deformation_graph[scale_index].root_vertex)
            else:
                shape_instance = PointDirectedGraph(
                    shape_instance.points,
                    self.deformation_graph[scale_index].adjacency_matrix)
        return shape_instance

    def view_shape_models_widget(self, n_parameters=5,
                                 parameters_bounds=(-3.0, 3.0),
                                 mode='multiple', figure_size=(10, 8)):
        r"""
        Visualize the shape model using an interactive widget.

        Parameters
        ----------
        n_parameters : `int`, optional
            The number of parameters that will be visible to change on the
            widget.
        parameters_bounds : ``(`float`, `float`)``, optional
            The minimum and maximum values of the parameters' sliders.
        mode : {``'single', 'multiple'``}, optional
            If 'multiple', then each parameter will have its own slider. If
            'single', then there will be a single slider and a drop down menu to
            select the parameters.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.
        """
        try:
            from menpowidgets import visualize_shape_model
            visualize_shape_model(self.shape_models, n_parameters=n_parameters,
                                  parameters_bounds=parameters_bounds,
                                  figure_size=figure_size, mode=mode)
        except:
            from menpo.visualize.base import MenpowidgetsMissingError
            raise MenpowidgetsMissingError()

    def view_shape_graph_widget(self, scale_index=-1, figure_size=(10, 8)):
        r"""
        Visualize the shape graph using an interactive widget.

        Parameters
        ----------
        scale_index : `int`, optional
            Defines the scale level, the shape model of which will be used.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.

        Raises
        ------
        ValueError
            Scale level {scale_index} uses a PCA shape model, so there is no
            graph
        """
        if self.shape_graph[scale_index] is not None:
            PointUndirectedGraph(
                self.shape_models[scale_index].mean().points,
                self.shape_graph[scale_index].adjacency_matrix).view_widget(
                figure_size=figure_size)
        else:
            raise ValueError("Scale level {} uses a PCA shape model, so there "
                             "is no graph".format(scale_index))

    def view_deformation_graph_widget(self, scale_index=-1,
                                      figure_size=(10, 8)):
        r"""
        Visualize the deformation graph using an interactive widget.

        Parameters
        ----------
        scale_index : `int`, optional
            Defines the scale level, the deformation model of which will be
            used.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.
        """
        if isinstance(self.deformation_graph[scale_index], Tree):
            PointTree(
                self.shape_models[scale_index].mean().points,
                self.deformation_graph[scale_index].adjacency_matrix,
                self.deformation_graph[scale_index].root_vertex).view_widget(
                figure_size=figure_size)
        else:
            PointDirectedGraph(
                self.shape_models[scale_index].mean().points,
                self.deformation_graph[scale_index].adjacency_matrix).\
                view_widget(figure_size=figure_size)

    def view_appearance_graph_widget(self, scale_index=-1, figure_size=(10, 8)):
        r"""
        Visualize the appearance graph using an interactive widget.

        Parameters
        ----------
        scale_index : `int`, optional
            Defines the scale level, the appearance model of which will be used.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.

        Raises
        ------
        ValueError
            Scale level {scale_index} uses a PCA appearance model, so there
            is no graph
        """
        if self.appearance_graph[scale_index] is not None:
            PointUndirectedGraph(
                self.shape_models[scale_index].mean().points,
                self.appearance_graph[scale_index].adjacency_matrix).\
                view_widget(figure_size=figure_size)
        else:
            raise ValueError("Scale level {} uses a PCA appearance model, "
                             "so there is no graph".format(scale_index))

    def view_deformation_model(self, scale_index=-1, n_std=2,
                               render_colour_bar=False, colour_map='jet',
                               image_view=True, figure_id=None,
                               new_figure=False, render_graph_lines=True,
                               graph_line_colour='b', graph_line_style='-',
                               graph_line_width=1., ellipse_line_colour='r',
                               ellipse_line_style='-', ellipse_line_width=1.,
                               render_markers=True, marker_style='o',
                               marker_size=20, marker_face_colour='k',
                               marker_edge_colour='k', marker_edge_width=1.,
                               render_axes=False,
                               axes_font_name='sans-serif', axes_font_size=10,
                               axes_font_style='normal',
                               axes_font_weight='normal', crop_proportion=0.1,
                               figure_size=(10, 8)):
        r"""
        Visualize the deformation model by plotting a Gaussian ellipsis per
        graph edge.

        Parameters
        ----------
        scale_index : `int`, optional
            Defines the scale level, the deformation model of which will be
            used.
        n_std : `float`, optional
            This defines the size of the ellipses in terms of number of standard
            deviations.
        render_colour_bar : `bool`, optional
            If ``True``, then the ellipses will be coloured based on their
            normalized standard deviations and a colour bar will also appear on
            the side. If ``False``, then all the ellipses will have the same
            colour.
        colour_map : `str`, optional
            A valid Matplotlib colour map. For more info, please refer to
            `matplotlib.cm`.
        image_view : `bool`, optional
            If ``True`` the ellipses will be rendered in the image coordinates
            system.
        figure_id : `object`, optional
            The id of the figure to be used.
        new_figure : `bool`, optional
            If ``True``, a new figure is created.
        render_graph_lines : `bool`, optional
            Defines whether to plot the graph's edges.
        graph_line_colour : See Below, optional
            The colour of the lines of the graph's edges.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        graph_line_style : ``{-, --, -., :}``, optional
            The style of the lines of the graph's edges.
        graph_line_width : `float`, optional
            The width of the lines of the graph's edges.
        ellipse_line_colour : See Below, optional
            The colour of the lines of the ellipses.
            Example options::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        ellipse_line_style : ``{-, --, -., :}``, optional
            The style of the lines of the ellipses.
        ellipse_line_width : `float`, optional
            The width of the lines of the ellipses.
        render_markers : `bool`, optional
            If ``True``, the centers of the ellipses will be rendered.
        marker_style : See Below, optional
            The style of the centers of the ellipses. Example options ::

                {., ,, o, v, ^, <, >, +, x, D, d, s, p, *, h, H, 1, 2, 3, 4, 8}

        marker_size : `int`, optional
            The size of the centers of the ellipses in points^2.
        marker_face_colour : See Below, optional
            The face (filling) colour of the centers of the ellipses.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_colour : See Below, optional
            The edge colour of the centers of the ellipses.
            Example options ::

                {r, g, b, c, m, k, w}
                or
                (3, ) ndarray

        marker_edge_width : `float`, optional
            The edge width of the centers of the ellipses.
        render_axes : `bool`, optional
            If ``True``, the axes will be rendered.
        axes_font_name : See Below, optional
            The font of the axes. Example options ::

                {serif, sans-serif, cursive, fantasy, monospace}

        axes_font_size : `int`, optional
            The font size of the axes.
        axes_font_style : ``{normal, italic, oblique}``, optional
            The font style of the axes.
        axes_font_weight : See Below, optional
            The font weight of the axes.
            Example options ::

                {ultralight, light, normal, regular, book, medium, roman,
                semibold,demibold, demi, bold, heavy, extra bold, black}

        crop_proportion : `float`, optional
            The proportion to be left around the centers' pointcloud.
        figure_size : (`float`, `float`) `tuple` or ``None`` optional
            The size of the figure in inches.
        """
        from menpo.visualize import plot_gaussian_ellipses

        mean_shape = self.shape_models[scale_index].mean().points
        deformation_graph = self.deformation_graph[scale_index]

        # get covariance matrices
        covariances = []
        means = []
        for e in range(deformation_graph.n_edges):
            # find vertices
            parent = deformation_graph.edges[e, 0]
            child = deformation_graph.edges[e, 1]

            # relative location mean
            means.append(mean_shape[child, :])

            # relative location cov
            s1 = -self.deformation_models[scale_index].precision[2 * child,
                                                                 2 * parent]
            s2 = -self.deformation_models[scale_index].precision[2 * child + 1,
                                                                 2 * parent + 1]
            s3 = -self.deformation_models[scale_index].precision[2 * child,
                                                                 2 * parent + 1]
            covariances.append(np.linalg.inv(np.array([[s1, s3], [s3, s2]])))

        # plot deformation graph
        if isinstance(deformation_graph, Tree):
            renderer = PointTree(
                mean_shape,
                deformation_graph.adjacency_matrix,
                deformation_graph.root_vertex).view(
                figure_id=figure_id, new_figure=new_figure,
                image_view=image_view, render_lines=render_graph_lines,
                line_colour=graph_line_colour, line_style=graph_line_style,
                line_width=graph_line_width, render_markers=render_markers,
                marker_style=marker_style, marker_size=marker_size,
                marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width, render_axes=render_axes,
                axes_font_name=axes_font_name, axes_font_size=axes_font_size,
                axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, figure_size=figure_size)
        else:
            renderer = PointDirectedGraph(
                mean_shape,
                deformation_graph.adjacency_matrix).view(
                figure_id=figure_id, new_figure=new_figure,
                image_view=image_view, render_lines=render_graph_lines,
                line_colour=graph_line_colour, line_style=graph_line_style,
                line_width=graph_line_width, render_markers=render_markers,
                marker_style=marker_style, marker_size=marker_size,
                marker_face_colour=marker_face_colour,
                marker_edge_colour=marker_edge_colour,
                marker_edge_width=marker_edge_width, render_axes=render_axes,
                axes_font_name=axes_font_name, axes_font_size=axes_font_size,
                axes_font_style=axes_font_style,
                axes_font_weight=axes_font_weight, figure_size=figure_size)

        # plot ellipses
        renderer = plot_gaussian_ellipses(
            covariances, means, n_std=n_std,
            render_colour_bar=render_colour_bar,
            colour_bar_label='Normalized Standard Deviation',
            colour_map=colour_map, figure_id=renderer.figure_id,
            new_figure=False, image_view=image_view,
            line_colour=ellipse_line_colour, line_style=ellipse_line_style,
            line_width=ellipse_line_width, render_markers=render_markers,
            marker_edge_colour=marker_edge_colour,
            marker_face_colour=marker_face_colour,
            marker_edge_width=marker_edge_width, marker_size=marker_size,
            marker_style=marker_style, render_axes=render_axes,
            axes_font_name=axes_font_name, axes_font_size=axes_font_size,
            axes_font_style=axes_font_style, axes_font_weight=axes_font_weight,
            crop_proportion=crop_proportion, figure_size=figure_size)

        return renderer

    def __str__(self):
        return self._str_title


def _compute_minimum_spanning_tree(shapes, root_vertex=0, prefix='',
                                   verbose=False):
    # initialize weights matrix
    n_vertices = shapes[0].n_points
    weights = np.zeros((n_vertices, n_vertices))

    # print progress if requested
    range1 = range(n_vertices-1)
    if verbose:
        range1 = print_progress(
            range1, end_with_newline=False,
            prefix='{}Deformation graph - Computing complete graph`s '
                   'weights'.format(prefix))

    # compute weights
    for i in range1:
        for j in range(i+1, n_vertices, 1):
            # create data matrix of edge
            diffs_x = [s.points[i, 0] - s.points[j, 0] for s in shapes]
            diffs_y = [s.points[i, 1] - s.points[j, 1] for s in shapes]
            coords = np.array([diffs_x, diffs_y])

            # compute mean and covariance
            m = np.mean(coords, axis=1)
            c = np.cov(coords)

            # get weight
            for im in range(len(shapes)):
                weights[i, j] += -np.log(multivariate_normal.pdf(coords[:, im],
                                                                 mean=m, cov=c))
            weights[j, i] = weights[i, j]

    # create undirected graph
    complete_graph = UndirectedGraph(weights)

    if verbose:
        print_dynamic('{}Deformation graph - Minimum spanning graph '
                      'computed.\n'.format(prefix))

    # compute minimum spanning graph
    return complete_graph.minimum_spanning_tree(root_vertex)
