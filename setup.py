from setuptools import find_packages, setup


def get_version_and_cmdclass(package_path):
    """Load version.py module without importing the whole package.

    Template code from miniver
    """
    import os
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("version", os.path.join(package_path, "_version.py"))
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.__version__, module.cmdclass


version, cmdclass = get_version_and_cmdclass("menpofit")


setup(
    name="menpofit",
    version=version,
    cmdclass=cmdclass,
    description="Menpo's image feature point localisation (AAMs, SDMs, CLMs)",
    author="The Menpo Development Team",
    author_email="james.booth08@imperial.ac.uk",
    packages=find_packages(),
    install_requires=["menpo>=0.9.0,<0.12.0", "scikit-learn>=0.16", "pandas>=0.24"],
    tests_require=["pytest>=5.0"],
)
