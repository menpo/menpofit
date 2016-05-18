from setuptools import setup, find_packages
import versioneer


setup(
      name='menpofit',
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass(),
      description="Menpo's image feature point localisation (AAMs, SDMs, "
                  "CLMs)",
      author='The Menpo Development Team',
      author_email='james.booth08@imperial.ac.uk',
      packages=find_packages(),
      install_requires=['menpo',
                        'scikit-learn'],
      tests_require=['nose', 'mock']
)
