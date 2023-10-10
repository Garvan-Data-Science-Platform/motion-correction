#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Pillow>=10.0.0',
    'numba>=0.57',
    'numba-progress>=1.1.0',
    'opencv-python>=4.8',
    "torch>=2.0.1",
    "numpy==1.23.3",
    "pyimof==1.0.0",
    "matplotlib==3.7.2",
    "numba>=0.58.0",
    "image-registration==0.2.6",
    "pytest>=7.4",
    "dipy>=1.7.0",
    "tqdm>=4.66",
    "scikit-image>=0.21",
    "sparse>=0.14.0",
    "numba_progress>=1.1.0",
    "cupy-cuda112",
]

test_requirements = ['pytest>=3', ]

setup(
    author="Timothy Kallady",
    author_email='t.kallady@garvan.org.au',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Motion correction algorithms for FLIM data.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='motion_correction',
    name='motion_correction',
    packages=find_packages(include=['motion_correction', 'motion_correction.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Garvan-Data-Science-Platformmotion_correction',
    version='0.1.1',
    zip_safe=False,
)
