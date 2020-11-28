## ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
install_requires=['torch' , 'opencv-python', 'matplotlib', 'numpy', 'PyYAML', 'Pillow','scipy', 'torchvision', 'tqdm', 'PyQt5', 'QtPy']

# fetch values from package.xml
setup_args = generate_distutils_setup(
	name='easynn_ros',         # Required
	version='2.0.0',  # Required

	packages=['your_package'],
	package_dir={'': 'src'},


	python_requires='>=3.5, <4',
	install_requires=install_requires,
)

for req in install_requires:
    install(req)

setup(**setup_args)
