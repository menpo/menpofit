import menpo

menpo_version = 'v' + menpo.__version__
menpo_rtd_base_url = 'http://menpo.readthedocs.org/en/{}/api/menpo/'.format(
    menpo_version)

xref_map = {
    'Copyable': ('url', menpo_rtd_base_url + 'base/Copyable.html'),
    'Vectorizable': ('url', menpo_rtd_base_url + 'base/Vectorizable.html'),
    'Image': ('url', menpo_rtd_base_url + 'image/Image.html'),
    'MaskedImage': ('url', menpo_rtd_base_url + 'image/MaskedImage.html'),
    'PointCloud': ('url', menpo_rtd_base_url + 'shape/PointCloud.html'),
    'Transform': ('url', menpo_rtd_base_url + 'transform/Transform.html'),
    'TransformChain': ('url', menpo_rtd_base_url + 'transform/TransformChain.html'),
    'Transformable': ('url', menpo_rtd_base_url + 'transform/Transformable.html')
}
