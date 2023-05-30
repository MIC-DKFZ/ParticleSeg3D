# ParticleSeg3D
ParticleSeg3D is an instance segmentation method that extracts individual particles from large micro CT images taken from mineral samples embedded in an epoxy matrix. It is built on the powerful nnU-Net framework, introduces a particle size normalization, and makes use of a border-core representation to enable instance segmentation.

## Features
- Robust instance segmentation of mineral particles in micro CT images
- Application of nnU-Net framework for reliable and scalable image processing
- Border-core representation for instance segmentation
- Particle size normalization to account for different mineral types
- Trained on a diverse set of particles from various materials and minerals
- Can be applied to a wide variety of particle types, without additional manual annotations or retraining

## Installation
Before running the code, you will need to install the required packages. Use the following command:

```
pip install -r requirements.txt
```