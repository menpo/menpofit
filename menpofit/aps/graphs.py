import numpy as np


def parse_deformation_graph(graph_type, n_points):
    if graph_type == 'full_multiple_gaussians':
        adjacency_array = np.array(_get_complete_directed_graph_edges(range(n_points)))
        root_vertex = None
    elif graph_type == 'full_multiple_gaussians_tri':
        adjacency_array = np.array(_get_complete_graph_edges(range(n_points)))
        root_vertex = None
    elif graph_type == 'chain_per_area_68':
        jaw = _get_chain_graph_edges(range(17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_chain_graph_edges(range(36, 42) + [36])
        leye = _get_chain_graph_edges(range(42, 48) + [42])
        mouth1 = _get_chain_graph_edges(range(48, 60) + [48])
        mouth2 = _get_chain_graph_edges(range(60, 68) + [60])
        edges = (jaw + rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye +
                 mouth1 + mouth2)
        adjacency_array = np.array(edges)
        root_vertex = None
    elif graph_type == 'chain_per_area_66':
        jaw = _get_chain_graph_edges(range(17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_chain_graph_edges(range(36, 42) + [36])
        leye = _get_chain_graph_edges(range(42, 48) + [42])
        mouth1 = _get_chain_graph_edges(range(48, 60) + [48])
        mouth2 = _get_chain_graph_edges(range(60, 66) + [60])
        edges = (jaw + rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye +
                 mouth1 + mouth2)
        adjacency_array = np.array(edges)
        root_vertex = None
    elif graph_type == 'chain_per_area_49':
        rbrow = _get_chain_graph_edges(range(5))
        lbrow = _get_chain_graph_edges(range(5, 10))
        nose1 = _get_chain_graph_edges(range(10, 14) + [16])
        nose2 = _get_chain_graph_edges(range(16, 13, -1))
        nose3 = _get_chain_graph_edges(range(16, 19))
        reye = _get_chain_graph_edges(range(19, 25) + [19])
        leye = _get_chain_graph_edges(range(25, 31) + [25])
        mouth1 = _get_chain_graph_edges(range(31, 43) + [31])
        mouth2 = _get_chain_graph_edges(range(43, 49) + [43])
        edges = (rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye + mouth1 +
                 mouth2)
        adjacency_array = np.array(edges)
        root_vertex = None
    elif graph_type == 'chain_per_area_unclosed_68':
        jaw = _get_chain_graph_edges(range(17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_chain_graph_edges(range(36, 42))
        leye = _get_chain_graph_edges(range(42, 48))
        mouth1 = _get_chain_graph_edges(range(48, 60))
        mouth2 = _get_chain_graph_edges(range(60, 68))
        edges = (jaw + rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye +
                 mouth1 + mouth2)
        adjacency_array = np.array(edges)
        root_vertex = None
    elif graph_type == 'chain_per_area_unclosed_66':
        jaw = _get_chain_graph_edges(range(17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_chain_graph_edges(range(36, 42))
        leye = _get_chain_graph_edges(range(42, 48))
        mouth1 = _get_chain_graph_edges(range(48, 60))
        mouth2 = _get_chain_graph_edges(range(60, 66))
        edges = (jaw + rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye +
                 mouth1 + mouth2)
        adjacency_array = np.array(edges)
        root_vertex = None
    elif graph_type == 'chain_per_area_unclosed_49':
        rbrow = _get_chain_graph_edges(range(5))
        lbrow = _get_chain_graph_edges(range(5, 10))
        nose1 = _get_chain_graph_edges(range(10, 14) + [16])
        nose2 = _get_chain_graph_edges(range(16, 13, -1))
        nose3 = _get_chain_graph_edges(range(16, 19))
        reye = _get_chain_graph_edges(range(19, 25))
        leye = _get_chain_graph_edges(range(25, 31))
        mouth1 = _get_chain_graph_edges(range(31, 43))
        mouth2 = _get_chain_graph_edges(range(43, 49))
        edges = (rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye + mouth1 +
                 mouth2)
        adjacency_array = np.array(edges)
        root_vertex = None
    elif graph_type == 'joan_graph_68':
        jaw = _get_chain_graph_edges(range(17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_complete_directed_graph_edges(range(36, 42))
        leye = _get_complete_directed_graph_edges(range(42, 48))
        mouth = _get_complete_directed_graph_edges(range(48, 68))
        edges = (jaw + [[36, 0], [17, 0], [45, 16], [26, 16]] + rbrow + lbrow +
                 [[19, 37], [24, 44]] + reye + leye + [[27, 39], [27, 42]] +
                 nose1 + nose2 + nose3 + mouth +
                 [[8, 57], [8, 66], [33, 51], [33, 62]])
        adjacency_array = np.array(edges)
        root_vertex = None
    elif graph_type == 'joan_graph_66':
        jaw = _get_chain_graph_edges(range(0, 17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_complete_directed_graph_edges(range(36, 42))
        leye = _get_complete_directed_graph_edges(range(42, 48))
        mouth = _get_complete_directed_graph_edges(range(48, 66))
        edges = (jaw + [[36, 0], [17, 0], [45, 16], [26, 16]] + rbrow + lbrow +
                 [[19, 37], [24, 44]] + reye + leye + [[27, 39], [27, 42]] +
                 nose1 + nose2 + nose3 + mouth +
                 [[8, 57], [8, 64], [33, 51], [33, 61]])
        adjacency_array = np.array(edges)
        root_vertex = None
    elif graph_type == 'joan_graph_49':
        rbrow = _get_chain_graph_edges(range(5))
        lbrow = _get_chain_graph_edges(range(5, 10))
        nose = (_get_chain_graph_edges([10, 11, 12, 13, 16]) +
                _get_chain_graph_edges([14, 15, 16, 17, 18]))
        reye = _get_complete_graph_edges(range(19, 25))
        leye = _get_complete_graph_edges(range(25, 31))
        mouth = _get_complete_graph_edges(range(31, 49))
        edges = (rbrow + lbrow + [[2, 20], [7, 27]] + reye + leye +
                 [[10, 22], [10, 25]] + nose + mouth + [[16, 34], [16, 44]])
        adjacency_array = np.array(edges)
        root_vertex = None
    elif graph_type == 'complete_and_chain_per_area_68':
        # define full for eyes and mouth
        jaw = _get_chain_graph_edges(range(0, 17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_complete_directed_graph_edges(range(36, 42))
        leye = _get_complete_directed_graph_edges(range(42, 48))
        mouth = _get_complete_directed_graph_edges(range(48, 66))
        edges = (jaw + rbrow + lbrow + reye + leye + nose1 + nose2 + nose3 +
                 mouth)
        adjacency_array = np.array(edges)
        root_vertex = None
    elif graph_type == 'complete_and_chain_per_area_66':
        # define full for eyes and mouth
        jaw = _get_chain_graph_edges(range(0, 17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_complete_directed_graph_edges(range(36, 42))
        leye = _get_complete_directed_graph_edges(range(42, 48))
        mouth = _get_complete_directed_graph_edges(range(48, 66))
        edges = (jaw + rbrow + lbrow + reye + leye + nose1 + nose2 + nose3 +
                 mouth)
        adjacency_array = np.array(edges)
        root_vertex = None
    elif graph_type == 'complete_and_chain_per_area_49':
        rbrow = _get_chain_graph_edges(range(5))
        lbrow = _get_chain_graph_edges(range(5, 10))
        nose = (_get_chain_graph_edges([10, 11, 12, 13, 16]) +
                _get_chain_graph_edges([14, 15, 16, 17, 18]))
        reye = _get_complete_graph_edges(range(19, 25))
        leye = _get_complete_graph_edges(range(25, 31))
        mouth = _get_complete_graph_edges(range(31, 49))
        adjacency_array = np.array(rbrow + lbrow + nose + reye + leye + mouth)
        root_vertex = None
    elif graph_type == 'mst_68':
        # MST 68
        adjacency_array = np.array(
            [[ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4], [ 4,  5], [ 5,  6],
             [ 6,  7], [ 7,  8], [ 8,  9], [ 8, 57], [ 9, 10], [57, 58],
             [57, 56], [57, 66], [10, 11], [58, 59], [56, 55], [66, 67],
             [66, 65], [11, 12], [65, 63], [12, 13], [63, 62], [63, 53],
             [13, 14], [62, 61], [62, 51], [53, 64], [14, 15], [61, 49],
             [51, 50], [51, 52], [51, 33], [64, 54], [15, 16], [49, 60],
             [33, 32], [33, 34], [33, 29], [60, 48], [32, 31], [34, 35],
             [29, 30], [29, 28], [28, 27], [27, 22], [27, 21], [22, 23],
             [21, 20], [23, 24], [20, 19], [24, 25], [19, 18], [25, 26],
             [25, 44], [18, 17], [18, 37], [44, 43], [44, 45], [37, 38],
             [45, 46], [38, 39], [46, 47], [39, 40], [47, 42], [40, 41],
             [41, 36]])
        root_vertex = 0
    elif graph_type == 'mst_66':
        adjacency_array = np.array(
            [[ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4], [ 4,  5], [ 5,  6],
             [ 6,  7], [ 7,  8], [ 8,  9], [ 8, 57], [ 9, 10], [57, 58],
             [57, 56], [57, 64], [10, 11], [58, 59], [56, 55], [64, 65],
             [64, 63], [11, 12], [63, 62], [12, 13], [62, 61], [62, 53],
             [13, 14], [61, 60], [61, 51], [53, 54], [14, 15], [60, 49],
             [51, 50], [51, 52], [51, 33], [15, 16], [49, 48], [33, 32],
             [33, 34], [33, 29], [32, 31], [34, 35], [29, 30], [29, 28],
             [28, 27], [27, 22], [27, 21], [22, 23], [21, 20], [23, 24],
             [20, 19], [24, 25], [19, 18], [25, 26], [25, 44], [18, 17],
             [18, 37], [44, 43], [44, 45], [37, 38], [45, 46], [38, 39],
             [46, 47], [39, 40], [47, 42], [40, 41], [41, 36]])
        root_vertex = 0
    elif graph_type == 'mst_49':
        adjacency_array = np.array(
            [[ 0,  1], [ 1,  2], [ 1, 20], [ 2,  3], [20, 21], [ 3,  4],
             [21, 22], [ 4, 10], [22, 23], [10, 11], [10,  5], [23, 24],
             [11, 12], [ 5,  6], [24, 19], [12, 13], [12, 16], [ 6,  7],
             [16, 15], [16, 17], [16, 34], [ 7,  8], [15, 14], [17, 18],
             [34, 33], [34, 44], [34, 35], [ 8,  9], [ 8, 27], [44, 43],
             [44, 45], [27, 26], [27, 28], [43, 32], [45, 46], [45, 36],
             [28, 29], [32, 31], [46, 47], [36, 37], [29, 30], [47, 48],
             [47, 40], [30, 25], [40, 41], [40, 39], [41, 42], [39, 38]])
        root_vertex = 0
    elif graph_type == 'mst_car_view0':
        adjacency_array = None
        root_vertex = 0
    elif graph_type == 'star_tree_68':
        # STAR 68
        adjacency_array = np.empty((67, 2), dtype=np.int32)
        for i in range(68):
            if i < 34:
                adjacency_array[i, 0] = 34
                adjacency_array[i, 1] = i
            elif i > 34:
                adjacency_array[i-1, 0] = 34
                adjacency_array[i-1, 1] = i
        root_vertex = 34
    elif graph_type == 'star_tree_66':
        # STAR 66
        adjacency_array = np.empty((65, 2), dtype=np.int32)
        for i in range(66):
            if i < 34:
                adjacency_array[i, 0] = 34
                adjacency_array[i, 1] = i
            elif i > 34:
                adjacency_array[i-1, 0] = 34
                adjacency_array[i-1, 1] = i
        root_vertex = 34
    elif graph_type == 'star_tree_51':
        # STAR 51
        adjacency_array = np.empty((50, 2), dtype=np.int32)
        for i in range(51):
            if i < 16:
                adjacency_array[i, 0] = 16
                adjacency_array[i, 1] = i
            elif i > 16:
                adjacency_array[i-1, 0] = 16
                adjacency_array[i-1, 1] = i
        root_vertex = 16
    elif graph_type == 'star_tree_49':
        # STAR 49
        adjacency_array = np.empty((48, 2), dtype=np.int32)
        for i in range(49):
            if i < 16:
                adjacency_array[i, 0] = 16
                adjacency_array[i, 1] = i
            elif i > 16:
                adjacency_array[i-1, 0] = 16
                adjacency_array[i-1, 1] = i
        root_vertex = 16
    else:
        raise ValueError('Invalid graph_deformation str provided')
    return adjacency_array, root_vertex


def parse_appearance_graph(graph_type, n_points):
    if graph_type == 'full_single_gaussian':
        # FULL
        adjacency_array = None
        gaussian_per_patch = False
    elif graph_type == 'full_yorgos_single_gaussian':
        # FULL
        adjacency_array = 'yorgos'
        gaussian_per_patch = False
    elif graph_type == 'full_multiple_gaussians':
        adjacency_array = np.array(_get_complete_graph_edges(range(n_points)))
        gaussian_per_patch = True
    elif graph_type == 'diagonal':
        # DIAGONAL
        adjacency_array = None
        gaussian_per_patch = True
    elif graph_type == 'chain_per_area_68':
        jaw = _get_chain_graph_edges(range(17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_chain_graph_edges(range(36, 42) + [36])
        leye = _get_chain_graph_edges(range(42, 48) + [42])
        mouth1 = _get_chain_graph_edges(range(48, 60) + [48])
        mouth2 = _get_chain_graph_edges(range(60, 68) + [60])
        edges = (jaw + rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye +
                 mouth1 + mouth2)
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'chain_per_area_66':
        jaw = _get_chain_graph_edges(range(17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_chain_graph_edges(range(36, 42) + [36])
        leye = _get_chain_graph_edges(range(42, 48) + [42])
        mouth1 = _get_chain_graph_edges(range(48, 60) + [48])
        mouth2 = _get_chain_graph_edges(range(60, 66) + [60])
        edges = (jaw + rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye +
                 mouth1 + mouth2)
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'chain_per_area_49':
        rbrow = _get_chain_graph_edges(range(5))
        lbrow = _get_chain_graph_edges(range(5, 10))
        nose1 = _get_chain_graph_edges(range(10, 14) + [16])
        nose2 = _get_chain_graph_edges(range(16, 13, -1))
        nose3 = _get_chain_graph_edges(range(16, 19))
        reye = _get_chain_graph_edges(range(19, 25) + [19])
        leye = _get_chain_graph_edges(range(25, 31) + [25])
        mouth1 = _get_chain_graph_edges(range(31, 43) + [31])
        mouth2 = _get_chain_graph_edges(range(43, 49) + [43])
        edges = (rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye + mouth1 +
                 mouth2)
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'joan_graph_68':
        # define full for eyes and mouth
        reye = _get_complete_graph_edges(range(36, 42))
        leye = _get_complete_graph_edges(range(42, 48))
        mouth = _get_complete_graph_edges(range(48, 68))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        jaw = _get_chain_graph_edges(range(0, 17))
        nose = (_get_chain_graph_edges([27, 28, 29, 30, 33]) +
                _get_chain_graph_edges([31, 32, 33, 34, 35]))
        edges = (jaw + [[36, 0], [17, 0], [45, 16], [26, 16]] + rbrow + lbrow +
                 [[19, 37], [24, 44]] + reye + leye + [[27, 39], [27, 42]] +
                 nose + mouth + [[8, 57], [8, 66], [33, 51], [33, 62]])
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'joan_graph_66':
        # define full for eyes and mouth
        reye = _get_complete_graph_edges(range(36, 42))
        leye = _get_complete_graph_edges(range(42, 48))
        mouth = _get_complete_graph_edges(range(48, 66))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        jaw = _get_chain_graph_edges(range(0, 17))
        nose = (_get_chain_graph_edges([27, 28, 29, 30, 33]) +
                _get_chain_graph_edges([31, 32, 33, 34, 35]))
        edges = (jaw + [[36, 0], [17, 0], [45, 16], [26, 16]] + rbrow + lbrow +
                 [[19, 37], [24, 44]] + reye + leye + [[27, 39], [27, 42]] +
                 nose + mouth + [[8, 57], [8, 64], [33, 51], [33, 61]])
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'joan_graph_49':
        rbrow = _get_chain_graph_edges(range(5))
        lbrow = _get_chain_graph_edges(range(5, 10))
        nose = (_get_chain_graph_edges([10, 11, 12, 13, 16]) +
                _get_chain_graph_edges([14, 15, 16, 17, 18]))
        reye = _get_complete_graph_edges(range(19, 25))
        leye = _get_complete_graph_edges(range(25, 31))
        mouth = _get_complete_graph_edges(range(31, 49))
        edges = (rbrow + lbrow + [[2, 20], [7, 27]] + reye + leye +
                 [[10, 22], [10, 25]] + nose + mouth + [[16, 34], [16, 44]])
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'complete_and_chain_per_area_68':
        jaw = _get_chain_graph_edges(range(0, 17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose = (_get_chain_graph_edges([27, 28, 29, 30, 33]) +
                _get_chain_graph_edges([31, 32, 33, 34, 35]))
        reye = _get_complete_graph_edges(range(36, 42))
        leye = _get_complete_graph_edges(range(42, 48))
        mouth = _get_complete_graph_edges(range(48, 68))
        adjacency_array = np.array(jaw + rbrow + lbrow + nose + reye + leye +
                                   mouth)
        gaussian_per_patch = True
    elif graph_type == 'complete_and_chain_per_area_66':
        jaw = _get_chain_graph_edges(range(0, 17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose = (_get_chain_graph_edges([27, 28, 29, 30, 33]) +
                _get_chain_graph_edges([31, 32, 33, 34, 35]))
        reye = _get_complete_graph_edges(range(36, 42))
        leye = _get_complete_graph_edges(range(42, 48))
        mouth = _get_complete_graph_edges(range(48, 66))
        adjacency_array = np.array(jaw + rbrow + lbrow + nose + reye + leye +
                                   mouth)
        gaussian_per_patch = True
    elif graph_type == 'complete_and_chain_per_area_49':
        rbrow = _get_chain_graph_edges(range(5))
        lbrow = _get_chain_graph_edges(range(5, 10))
        nose = (_get_chain_graph_edges([10, 11, 12, 13, 16]) +
                _get_chain_graph_edges([14, 15, 16, 17, 18]))
        reye = _get_complete_graph_edges(range(19, 25))
        leye = _get_complete_graph_edges(range(25, 31))
        mouth = _get_complete_graph_edges(range(31, 49))
        adjacency_array = np.array(rbrow + lbrow + nose + reye + leye + mouth)
        gaussian_per_patch = True
    elif graph_type == 'mst_68':
        # MST 68
        adjacency_array = np.array(
            [[ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4], [ 4,  5], [ 5,  6],
             [ 6,  7], [ 7,  8], [ 8,  9], [ 8, 57], [ 9, 10], [57, 58],
             [57, 56], [57, 66], [10, 11], [58, 59], [56, 55], [66, 67],
             [66, 65], [11, 12], [65, 63], [12, 13], [63, 62], [63, 53],
             [13, 14], [62, 61], [62, 51], [53, 64], [14, 15], [61, 49],
             [51, 50], [51, 52], [51, 33], [64, 54], [15, 16], [49, 60],
             [33, 32], [33, 34], [33, 29], [60, 48], [32, 31], [34, 35],
             [29, 30], [29, 28], [28, 27], [27, 22], [27, 21], [22, 23],
             [21, 20], [23, 24], [20, 19], [24, 25], [19, 18], [25, 26],
             [25, 44], [18, 17], [18, 37], [44, 43], [44, 45], [37, 38],
             [45, 46], [38, 39], [46, 47], [39, 40], [47, 42], [40, 41],
             [41, 36]])
        gaussian_per_patch = True
    elif graph_type == 'mst_66':
        adjacency_array = np.array(
            [[ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4], [ 4,  5], [ 5,  6],
             [ 6,  7], [ 7,  8], [ 8,  9], [ 8, 57], [ 9, 10], [57, 58],
             [57, 56], [57, 64], [10, 11], [58, 59], [56, 55], [64, 65],
             [64, 63], [11, 12], [63, 62], [12, 13], [62, 61], [62, 53],
             [13, 14], [61, 60], [61, 51], [53, 54], [14, 15], [60, 49],
             [51, 50], [51, 52], [51, 33], [15, 16], [49, 48], [33, 32],
             [33, 34], [33, 29], [32, 31], [34, 35], [29, 30], [29, 28],
             [28, 27], [27, 22], [27, 21], [22, 23], [21, 20], [23, 24],
             [20, 19], [24, 25], [19, 18], [25, 26], [25, 44], [18, 17],
             [18, 37], [44, 43], [44, 45], [37, 38], [45, 46], [38, 39],
             [46, 47], [39, 40], [47, 42], [40, 41], [41, 36]])
        gaussian_per_patch = True
    elif graph_type == 'mst_49':
        adjacency_array = np.array(
            [[ 0,  1], [ 1,  2], [ 1, 20], [ 2,  3], [20, 21], [ 3,  4],
             [21, 22], [ 4, 10], [22, 23], [10, 11], [10,  5], [23, 24],
             [11, 12], [ 5,  6], [24, 19], [12, 13], [12, 16], [ 6,  7],
             [16, 15], [16, 17], [16, 34], [ 7,  8], [15, 14], [17, 18],
             [34, 33], [34, 44], [34, 35], [ 8,  9], [ 8, 27], [44, 43],
             [44, 45], [27, 26], [27, 28], [43, 32], [45, 46], [45, 36],
             [28, 29], [32, 31], [46, 47], [36, 37], [29, 30], [47, 48],
             [47, 40], [30, 25], [40, 41], [40, 39], [41, 42], [39, 38]])
        gaussian_per_patch = True
    elif graph_type == 'star_tree_68':
        # STAR 68
        adjacency_array = np.empty((67, 2), dtype=np.int32)
        for i in range(68):
            if i < 34:
                adjacency_array[i, 0] = 34
                adjacency_array[i, 1] = i
            elif i > 34:
                adjacency_array[i-1, 0] = 34
                adjacency_array[i-1, 1] = i
        gaussian_per_patch = True
    elif graph_type == 'star_tree_66':
        # STAR 68
        adjacency_array = np.empty((65, 2), dtype=np.int32)
        for i in range(66):
            if i < 34:
                adjacency_array[i, 0] = 34
                adjacency_array[i, 1] = i
            elif i > 34:
                adjacency_array[i-1, 0] = 34
                adjacency_array[i-1, 1] = i
        gaussian_per_patch = True
    elif graph_type == 'star_tree_51':
        # STAR 51
        adjacency_array = np.empty((50, 2), dtype=np.int32)
        for i in range(51):
            if i < 16:
                adjacency_array[i, 0] = 16
                adjacency_array[i, 1] = i
            elif i > 16:
                adjacency_array[i-1, 0] = 16
                adjacency_array[i-1, 1] = i
        gaussian_per_patch = True
    elif graph_type == 'star_tree_49':
        # STAR 49
        adjacency_array = np.empty((48, 2), dtype=np.int32)
        for i in range(49):
            if i < 16:
                adjacency_array[i, 0] = 16
                adjacency_array[i, 1] = i
            elif i > 16:
                adjacency_array[i-1, 0] = 16
                adjacency_array[i-1, 1] = i
        gaussian_per_patch = True
    else:
        raise ValueError('Invalid graph_appearance str provided')
    return adjacency_array, gaussian_per_patch


def parse_shape_graph(graph_type, n_points):
    if graph_type == 'full_single_gaussian':
        # FULL
        adjacency_array = None
        gaussian_per_patch = False
    elif graph_type == 'full_yorgos_single_gaussian':
        # FULL
        adjacency_array = 'yorgos'
        gaussian_per_patch = False
    elif graph_type == 'full_multiple_gaussians':
        adjacency_array = np.array(_get_complete_graph_edges(range(n_points)))
        gaussian_per_patch = True
    elif graph_type == 'diagonal':
        adjacency_array = np.array(_get_diagonal_graph_edges(range(n_points)))
        gaussian_per_patch = True
    elif graph_type == 'chain_per_area_68':
        jaw = _get_chain_graph_edges(range(17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_chain_graph_edges(range(36, 42) + [36])
        leye = _get_chain_graph_edges(range(42, 48) + [42])
        mouth1 = _get_chain_graph_edges(range(48, 60) + [48])
        mouth2 = _get_chain_graph_edges(range(60, 68) + [60])
        edges = (jaw + rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye +
                 mouth1 + mouth2)
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'chain_per_area_66':
        jaw = _get_chain_graph_edges(range(17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose1 = _get_chain_graph_edges(range(27, 31) + [33])
        nose2 = _get_chain_graph_edges(range(33, 30, -1))
        nose3 = _get_chain_graph_edges(range(33, 36))
        reye = _get_chain_graph_edges(range(36, 42) + [36])
        leye = _get_chain_graph_edges(range(42, 48) + [42])
        mouth1 = _get_chain_graph_edges(range(48, 60) + [48])
        mouth2 = _get_chain_graph_edges(range(60, 66) + [60])
        edges = (jaw + rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye +
                 mouth1 + mouth2)
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'chain_per_area_49':
        rbrow = _get_chain_graph_edges(range(5))
        lbrow = _get_chain_graph_edges(range(5, 10))
        nose1 = _get_chain_graph_edges(range(10, 14) + [16])
        nose2 = _get_chain_graph_edges(range(16, 13, -1))
        nose3 = _get_chain_graph_edges(range(16, 19))
        reye = _get_chain_graph_edges(range(19, 25) + [19])
        leye = _get_chain_graph_edges(range(25, 31) + [25])
        mouth1 = _get_chain_graph_edges(range(31, 43) + [31])
        mouth2 = _get_chain_graph_edges(range(43, 49) + [43])
        edges = (rbrow + lbrow + nose1 + nose2 + nose3 + reye + leye + mouth1 +
                 mouth2)
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'joan_graph_68':
        # define full for eyes and mouth
        reye = _get_complete_graph_edges(range(36, 42))
        leye = _get_complete_graph_edges(range(42, 48))
        mouth = _get_complete_graph_edges(range(48, 68))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        jaw = _get_chain_graph_edges(range(0, 17))
        nose = (_get_chain_graph_edges([27, 28, 29, 30, 33]) +
                _get_chain_graph_edges([31, 32, 33, 34, 35]))
        edges = (jaw + [[36, 0], [17, 0], [45, 16], [26, 16]] + rbrow + lbrow +
                 [[19, 37], [24, 44]] + reye + leye + [[27, 39], [27, 42]] +
                 nose + mouth + [[8, 57], [8, 66], [33, 51], [33, 62]])
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'joan_graph_66':
        # define full for eyes and mouth
        reye = _get_complete_graph_edges(range(36, 42))
        leye = _get_complete_graph_edges(range(42, 48))
        mouth = _get_complete_graph_edges(range(48, 66))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        jaw = _get_chain_graph_edges(range(0, 17))
        nose = (_get_chain_graph_edges([27, 28, 29, 30, 33]) +
                _get_chain_graph_edges([31, 32, 33, 34, 35]))
        edges = (jaw + [[36, 0], [17, 0], [45, 16], [26, 16]] + rbrow + lbrow +
                 [[19, 37], [24, 44]] + reye + leye + [[27, 39], [27, 42]] +
                 nose + mouth + [[8, 57], [8, 64], [33, 51], [33, 61]])
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'joan_graph_49':
        rbrow = _get_chain_graph_edges(range(5))
        lbrow = _get_chain_graph_edges(range(5, 10))
        nose = (_get_chain_graph_edges([10, 11, 12, 13, 16]) +
                _get_chain_graph_edges([14, 15, 16, 17, 18]))
        reye = _get_complete_graph_edges(range(19, 25))
        leye = _get_complete_graph_edges(range(25, 31))
        mouth = _get_complete_graph_edges(range(31, 49))
        edges = (rbrow + lbrow + [[2, 20], [7, 27]] + reye + leye +
                 [[10, 22], [10, 25]] + nose + mouth + [[16, 34], [16, 44]])
        adjacency_array = np.array(edges)
        gaussian_per_patch = True
    elif graph_type == 'complete_and_chain_per_area_68':
        jaw = _get_chain_graph_edges(range(0, 17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose = (_get_chain_graph_edges([27, 28, 29, 30, 33]) +
                _get_chain_graph_edges([31, 32, 33, 34, 35]))
        reye = _get_complete_graph_edges(range(36, 42))
        leye = _get_complete_graph_edges(range(42, 48))
        mouth = _get_complete_graph_edges(range(48, 68))
        adjacency_array = np.array(jaw + rbrow + lbrow + nose + reye + leye +
                                   mouth)
        gaussian_per_patch = True
    elif graph_type == 'complete_and_chain_per_area_66':
        jaw = _get_chain_graph_edges(range(0, 17))
        rbrow = _get_chain_graph_edges(range(17, 22))
        lbrow = _get_chain_graph_edges(range(22, 27))
        nose = (_get_chain_graph_edges([27, 28, 29, 30, 33]) +
                _get_chain_graph_edges([31, 32, 33, 34, 35]))
        reye = _get_complete_graph_edges(range(36, 42))
        leye = _get_complete_graph_edges(range(42, 48))
        mouth = _get_complete_graph_edges(range(48, 66))
        adjacency_array = np.array(jaw + rbrow + lbrow + nose + reye + leye +
                                   mouth)
        gaussian_per_patch = True
    elif graph_type == 'complete_and_chain_per_area_49':
        rbrow = _get_chain_graph_edges(range(5))
        lbrow = _get_chain_graph_edges(range(5, 10))
        nose = (_get_chain_graph_edges([10, 11, 12, 13, 16]) +
                _get_chain_graph_edges([14, 15, 16, 17, 18]))
        reye = _get_complete_graph_edges(range(19, 25))
        leye = _get_complete_graph_edges(range(25, 31))
        mouth = _get_complete_graph_edges(range(31, 49))
        adjacency_array = np.array(rbrow + lbrow + nose + reye + leye + mouth)
        gaussian_per_patch = True
    elif graph_type == 'mst_68':
        # MST 68
        adjacency_array = np.array(
            [[ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4], [ 4,  5], [ 5,  6],
             [ 6,  7], [ 7,  8], [ 8,  9], [ 8, 57], [ 9, 10], [57, 58],
             [57, 56], [57, 66], [10, 11], [58, 59], [56, 55], [66, 67],
             [66, 65], [11, 12], [65, 63], [12, 13], [63, 62], [63, 53],
             [13, 14], [62, 61], [62, 51], [53, 64], [14, 15], [61, 49],
             [51, 50], [51, 52], [51, 33], [64, 54], [15, 16], [49, 60],
             [33, 32], [33, 34], [33, 29], [60, 48], [32, 31], [34, 35],
             [29, 30], [29, 28], [28, 27], [27, 22], [27, 21], [22, 23],
             [21, 20], [23, 24], [20, 19], [24, 25], [19, 18], [25, 26],
             [25, 44], [18, 17], [18, 37], [44, 43], [44, 45], [37, 38],
             [45, 46], [38, 39], [46, 47], [39, 40], [47, 42], [40, 41],
             [41, 36]])
        gaussian_per_patch = True
    elif graph_type == 'mst_66':
        adjacency_array = np.array(
            [[ 0,  1], [ 1,  2], [ 2,  3], [ 3,  4], [ 4,  5], [ 5,  6],
             [ 6,  7], [ 7,  8], [ 8,  9], [ 8, 57], [ 9, 10], [57, 58],
             [57, 56], [57, 64], [10, 11], [58, 59], [56, 55], [64, 65],
             [64, 63], [11, 12], [63, 62], [12, 13], [62, 61], [62, 53],
             [13, 14], [61, 60], [61, 51], [53, 54], [14, 15], [60, 49],
             [51, 50], [51, 52], [51, 33], [15, 16], [49, 48], [33, 32],
             [33, 34], [33, 29], [32, 31], [34, 35], [29, 30], [29, 28],
             [28, 27], [27, 22], [27, 21], [22, 23], [21, 20], [23, 24],
             [20, 19], [24, 25], [19, 18], [25, 26], [25, 44], [18, 17],
             [18, 37], [44, 43], [44, 45], [37, 38], [45, 46], [38, 39],
             [46, 47], [39, 40], [47, 42], [40, 41], [41, 36]])
        gaussian_per_patch = True
    elif graph_type == 'mst_49':
        adjacency_array = np.array(
            [[ 0,  1], [ 1,  2], [ 1, 20], [ 2,  3], [20, 21], [ 3,  4],
             [21, 22], [ 4, 10], [22, 23], [10, 11], [10,  5], [23, 24],
             [11, 12], [ 5,  6], [24, 19], [12, 13], [12, 16], [ 6,  7],
             [16, 15], [16, 17], [16, 34], [ 7,  8], [15, 14], [17, 18],
             [34, 33], [34, 44], [34, 35], [ 8,  9], [ 8, 27], [44, 43],
             [44, 45], [27, 26], [27, 28], [43, 32], [45, 46], [45, 36],
             [28, 29], [32, 31], [46, 47], [36, 37], [29, 30], [47, 48],
             [47, 40], [30, 25], [40, 41], [40, 39], [41, 42], [39, 38]])
        gaussian_per_patch = True
    elif graph_type == 'star_tree_68':
        # STAR 68
        adjacency_array = np.empty((67, 2), dtype=np.int32)
        for i in range(68):
            if i < 34:
                adjacency_array[i, 0] = 34
                adjacency_array[i, 1] = i
            elif i > 34:
                adjacency_array[i-1, 0] = 34
                adjacency_array[i-1, 1] = i
        gaussian_per_patch = True
    elif graph_type == 'star_tree_66':
        # STAR 68
        adjacency_array = np.empty((65, 2), dtype=np.int32)
        for i in range(66):
            if i < 34:
                adjacency_array[i, 0] = 34
                adjacency_array[i, 1] = i
            elif i > 34:
                adjacency_array[i-1, 0] = 34
                adjacency_array[i-1, 1] = i
        gaussian_per_patch = True
    elif graph_type == 'star_tree_51':
        # STAR 51
        adjacency_array = np.empty((50, 2), dtype=np.int32)
        for i in range(51):
            if i < 16:
                adjacency_array[i, 0] = 16
                adjacency_array[i, 1] = i
            elif i > 16:
                adjacency_array[i-1, 0] = 16
                adjacency_array[i-1, 1] = i
        gaussian_per_patch = True
    elif graph_type == 'star_tree_49':
        # STAR 49
        adjacency_array = np.empty((48, 2), dtype=np.int32)
        for i in range(49):
            if i < 16:
                adjacency_array[i, 0] = 16
                adjacency_array[i, 1] = i
            elif i > 16:
                adjacency_array[i-1, 0] = 16
                adjacency_array[i-1, 1] = i
        gaussian_per_patch = True
    else:
        raise ValueError('Invalid graph_appearance str provided')
    return adjacency_array, gaussian_per_patch


def _get_complete_graph_edges(vertices_list):
    n_vertices = len(vertices_list)
    edges = []
    for i in range(n_vertices-1):
        k = i + 1
        for j in range(k, n_vertices, 1):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            edges.append([v1, v2])
    return edges


def _get_chain_graph_edges(vertices_list):
    n_vertices = len(vertices_list)
    edges = []
    for i in range(n_vertices-1):
        k = i + 1
        v1 = vertices_list[i]
        v2 = vertices_list[k]
        edges.append([v1, v2])
    return edges


def _get_complete_directed_graph_edges(vertices_list):
    n_vertices = len(vertices_list)
    edges = []
    for i in range(n_vertices-1):
        k = i + 1
        for j in range(k, n_vertices, 1):
            v1 = vertices_list[i]
            v2 = vertices_list[j]
            edges.append([v1, v2])
            edges.append([v2, v1])
    return edges


def _get_diagonal_graph_edges(vertices_list):
    edges = np.empty((len(vertices_list), 2), dtype=np.int)
    for i, v in enumerate(vertices_list):
        edges[i, 0] = v
        edges[i, 1] = v
    return edges
