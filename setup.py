#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "tqdm",
    "SimpleITK",
    "zarr",
    "GeodisTK",
    "connected-components-3d",
    "numpy-indexed",
    "scikit-image",
    "acvl-utils",
    "natsort",
    "pytorch_lightning<=1.9.5",
    "nnUNet @ git+https://github.com/MIC-DKFZ/nnUNet.git@ParticleSeg3D", # @ParticleSeg3D#egg=nnunetv1_particleseg3d
]

test_requirements = [ ]

setup(
    author="Karol Gotkowski",
    author_email='karol.gotkowski@dkfz.de',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Scalable, out-of-the box segmentation of individual particles from mineral samples acquired with micro CT",
    entry_points={
        'console_scripts': [
            'ps3d_inference = particleseg3d.inference.inference:main',
            'ps3d_tiff2zarr = particleseg3d.conversion.tiff2zarr:main',
            'ps3d_zarr2tiff = particleseg3d.conversion.zarr2tiff:main',
            'ps3d_train_preprocess = particleseg3d.train.preprocess:main',
        ],
    },
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='particleseg3d',
    name='particleseg3d',
    packages=find_packages(include=['particleseg3d', 'particleseg3d.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/Karol-G/particleseg3d',
    version='0.1.0',
    zip_safe=False,
)
