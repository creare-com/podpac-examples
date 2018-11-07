""" podpac module"""

# Always perfer setuptools over distutils
import sys
from setuptools import setup, find_packages

# get version
sys.path.insert(0, 'podpac_examples')
import version
__version__ = version.version()

install_requires = {}
if sys.version_info.major == 2:
    install_requires += ['future>=0.16']

extras_require = {}

all_reqs = []
for key, val in extras_require.items():
    all_reqs += val
extras_require['all'] = all_reqs

setup(
    # ext_modules=None,
    name='podpac_examples',

    version=__version__,

    description="Pipeline for Observational Data Processing, Analysis, and Collaboration, example files",
    author='Creare',
    url="https://github.com/creare-com/podpac_examples",
    license="APACHE 2.0",
    classifiers=[
        # How mature is this project? Common values are
        # 3 - Alpha
        # 4 - Beta
        # 5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: GIS',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: both',
    ],
    packages=find_packages(),
    install_requires=install_requires,
    extras_require=extras_require,
)
