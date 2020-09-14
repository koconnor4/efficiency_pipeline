from setuptools import setup
import os,glob,warnings,sys,fnmatch,subprocess
from setuptools.command.test import test as TestCommand
from distutils.core import setup
import numpy.distutils.misc_util


if sys.version_info < (3,0):
    sys.exit('Sorry, Python 2 is not supported')

class efficiency_pipelinetest(TestCommand):

   def run_tests(self):
       import efficiency_pipeline
       errno = efficiency_pipeline.test()
       efficiency_pipeline.test_efficiency_pipeline()
       sys.exit(errno)

AUTHOR = 'Kyle OConnor'
AUTHOR_EMAIL = 'oconnorf@email.sc.edu'
VERSION = '0.0.1'
LICENSE = ''
URL = ''



def recursive_glob(basedir, pattern):
    matches = []
    for root, dirnames, filenames in os.walk(basedir):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches

PACKAGENAME='efficiency_pipeline'


# Add the project-global data
data_files = []
for dataFolderName in ['08.17']:
  pkgdatadir = os.path.join(PACKAGENAME, dataFolderName)
  data_files.extend(recursive_glob(pkgdatadir, '*'))

data_files = [f[len(PACKAGENAME)+1:] for f in data_files]


setup(
    name=PACKAGENAME,
    cmdclass={'test': efficiency_pipelinetest},
    setup_requires='numpy',
    install_requires=['matplotlib', 'numpy', 'scipy', 'astropy', 'astroquery', 'photutils', 'requests', 'pytest-astropy'],
    packages=[PACKAGENAME],
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    license=LICENSE,
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
    package_data={PACKAGENAME:data_files},
    include_package_data=True
)
