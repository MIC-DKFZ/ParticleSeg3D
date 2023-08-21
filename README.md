# ParticleSeg3D

[![License Apache Software License 2.0](https://img.shields.io/pypi/l/ParticleSeg3D.svg?color=green)](https://github.com/Karol-G/ParticleSeg3D/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ParticleSeg3D.svg?color=green)](https://pypi.org/project/ParticleSeg3D)
[![Python Version](https://img.shields.io/pypi/pyversions/ParticleSeg3D.svg?color=green)](https://python.org)
[![codecov](https://codecov.io/gh/Karol-G/ParticleSeg3D/branch/main/graph/badge.svg)](https://codecov.io/gh/Karol-G/ParticleSeg3D)

[ParticleSeg3D](https://arxiv.org/abs/2301.13319) is an instance segmentation method that extracts individual particles from large micro CT images taken from mineral samples embedded in an epoxy matrix. It is built on the powerful nnU-Net framework, introduces a particle size normalization, and makes use of a border-core representation to enable instance segmentation.

<p align="center">
  <img width="500" src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDVjOThmZGU3ZmM1Yzg0YzFlNDQyYzViOWIyODdlYTE1ZmNjM2FiNSZlcD12MV9pbnRlcm5hbF9naWZzX2dpZklkJmN0PWc/GxoBNxpCt79Rxt0Ezj/giphy.gif">
</p>

## Features
- Robust instance segmentation of mineral particles in micro CT images
- Application of nnU-Net framework for reliable and scalable image processing
- Border-core representation for instance segmentation
- Particle size normalization to account for different mineral types
- Trained on a diverse set of particles from various materials and minerals
- Can be applied to a wide variety of particle types, without additional manual annotations or retraining

## Installation
You can install `ParticleSeg3D` via [pip](https://pypi.org/project/ParticleSeg3D/):

    pip install ParticleSeg3D

You should now have the ParticleSeg3D package installed in your Python environment, and you'll be able to use all ParticleSeg3D commands from anywhere on your system.

If you intend to train ParticleSeg3D on new data, you will need to additionally install a modified version of the nnU-Net V1:
```cmd
pip install git+https://github.com/MIC-DKFZ/nnUNet.git@ParticleSeg3D
```

## Dataset

The sample dataset consisting of the whole CT images and the patch dataset with extracted patches from these samples alongside their respective instance segmentations can be found [here](https://syncandshare.desy.de/index.php/s/wjiDQ49KangiPj5).

## Usage - Inference

### Model download
ParticleSeg3D requires a trained model in order to run inference. The trained model can be downloaded [here](https://syncandshare.desy.de/index.php/s/id9D9pkATrFw65s). After downloading the weights, the weights need to be unpacked and saved at a location of your choosing.

### Conversion to Zarr
To run inference on an image using ParticleSeg3D, the image must first be converted into the Zarr format. The Zarr format suits our purposes well as it is designed for very large N-dimensional images. In case of a series of TIFF image files, this conversion can be accomplished using the following command from anywhere on the system:
```cmd
ps3d_tiff2zarr -i /path/to/input -o /path/to/output
```

Here's a breakdown of relevant arguments you should provide:
- '-i', '--input': Required. Absolute input path to the folder that contains the TIFF image slices that should be converted to a Zarr image.
- '-o', '--output': Required. Absolute output path to the folder that should be used to save the Zarr image.

### Metadata preparation
ParticleSeg3D requires the image spacing and a rough mean particle diameter size in millimeter of each image that should be inferenced. 
This information needs to be provided in the form of a metadata.json as shown in this example:
```json
{
    "Ore1_Zone3_Concentrate": {
        "spacing": 0.01,
        "particle_size": 0.29292
    },
    "Recycling1": {
        "spacing": 0.011,
        "particle_size": 0.5082
    },
    "Ore2_PS850_VS10": {
        "spacing": 0.01,
        "particle_size": 1.2874
    },
    "Ore5": {
        "spacing": 0.0055,
        "particle_size": 0.2296
    },
    ...
}
```


### Inference
You can run inference on Zarr images from anywhere on the system using the ps3d_inference command. The Zarr images need to be located in a folder named 'images' and the 'metadata.json' needs to be placed next to the folder such that the folder structure looks like this:
```
.
├── metadata.json
└── images
    ├── Ore1_Zone3_Concentrate.zarr
    ├── Recycling1.zarr
    ├── Ore2_PS850_VS10.zarr
    ├── Ore5.zarr
    └── ...
```


Here's an example of how to use the command:
```cmd
ps3d_inference -i /path/to/input -o /path/to/output -m /path/to/model
```

Here's a breakdown of relevant arguments you should provide:

- '-i', '--input': Required. Absolute input path to the base folder containing the dataset. The dataset should be structured with 'images' directory and metadata.json.
- '-o', '--output': Required. Absolute output path to the save folder.
- '-m', '--model': Required. Absolute path to the model directory.
- '-n', '--name': Optional. The name(s) without extension of the image(s) that should be used for inference. Multiple names must be separated by spaces.

### Conversion from Zarr
Zarr images or Zarr predictions can be converted to TIFF  using the following command from anywhere on the system:
```cmd
ps3d_zarr2tiff -i /path/to/input -o /path/to/output
```

Here's a breakdown of relevant arguments you should provide:
- '-i', '--input': Required. Absolute input path to the folder that contains the TIFF image slices that should be converted to a Zarr image.
- '-o', '--output': Required. Absolute output path to the folder that should be used to save the Zarr image.


## Usage - Training

### Conversion to NIFTI
To train a new ParticleSeg3D model on new training images, the training images must first be converted into the NIFTI format. The NIFTI format is required as input format by the nnU-Net. In case of a series of TIFF image files, this conversion can be accomplished using the following command from anywhere on the system:
```cmd
ps3d_tiff2nifti -i /path/to/input -o /path/to/output -s 0.1 0.1 0.1
```

Here's a breakdown of relevant arguments you should provide:
- '-i', '--input': Required. Absolute input path to the folder that contains the TIFF image slices that should be converted to a Zarr image.
- '-o', '--output': Required. Absolute output path to the folder that should be used to save the Zarr image.
- '-s', '--spacing': Required. The image spacing given as three numbers separate by spaces.

### Metadata preparation
ParticleSeg3D requires the image spacing and a rough mean particle diameter size in millimeter of each image that should be used for training. 
This information needs to be provided in the form of a metadata.json as shown in this example:
```json
{
    "Ore1_Zone3_Concentrate": {
        "spacing": 0.01,
        "particle_size": 0.29292
    },
    "Recycling1": {
        "spacing": 0.011,
        "particle_size": 0.5082
    },
    "Ore2_PS850_VS10": {
        "spacing": 0.01,
        "particle_size": 1.2874
    },
    "Ore5": {
        "spacing": 0.0055,
        "particle_size": 0.2296
    },
    ...
}
```

### Z-Score preparation
ParticleSeg3D performs Z-score intensity normalization and thus requires the global mean and standard deviation of the entire dataset. This can either be exactly computed over all voxel of all images combined of estimated by randomly sampling a subset of voxels from all images. The second option might be more convinient on larger images.

### Dataset preprocessing
The NIFTI images and reference instance segmentations need to be preprocessed into the for the nnU-Net expected dataset format. The  NIFTI images need to be located in a folder named 'images', the NIFTI instance segmentations in a folder named 'instance_seg' and the 'metadata.json' needs to be placed next to both folders. Further, images and their respective instance segmentations should have the same name.The folder structure should look like this:
```
.
├── metadata.json
├── images
│   ├── Ore1_Zone3_Concentrate.zarr
│   ├── Recycling1.zarr
│   ├── Ore2_PS850_VS10.zarr
│   ├── Ore5.zarr
│   └── ...
└── instance_seg
    ├── Ore1_Zone3_Concentrate.zarr
    ├── Recycling1.zarr
    ├── Ore2_PS850_VS10.zarr
    ├── Ore5.zarr
    └── ...
```

The dataset can then be preprocessed into nnU-Net format with the following command:
```cmd
ps3d_train_preprocess -i /path/to/input -o /path/to/output -z 0.12345 0.6789
```

Here's a breakdown of relevant arguments you should provide:

- '-i', '--input': Required. Absolute input path to the base folder that contains the dataset structured in the form of the directories 'images' and 'instance_seg' and the file metadata.json.
- '-o', '--output': Required. Absolute output path to the preprocessed dataset directory.
- '-z', '--zscore': Required. The z-score used for intensity normalization.

### nnU-Net training

After the dataset has been preprocessed the training of the nnU-Net model can commence. In order to this, it is best to follow the instructions from the official nnU-Net V1 [documentation](https://github.com/MIC-DKFZ/nnUNet/tree/nnunetv1). Once the training finished the trained model can be used for inference on new images.

## License

Distributed under the terms of the [Apache Software License 2.0](http://www.apache.org/licenses/LICENSE-2.0) license,
"ParticleSeg3D" is free and open source software

# Citations

If you are using ParticleSeg3D for your article, please consider citing our paper:

```
@misc{gotkowski2023work,
      title={[Work in progress] Scalable, out-of-the box segmentation of individual particles from mineral samples acquired with micro CT}, 
      author={Karol Gotkowski and Shuvam Gupta and Jose R. A. Godinho and Camila G. S. Tochtrop and Klaus H. Maier-Hein and Fabian Isensee},
      year={2023},
      eprint={2301.13319},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgements
<img src="https://github.com/MIC-DKFZ/ParticleSeg3D/raw/main/HI_Logo.png" height="100px" />

<img src="https://github.com/MIC-DKFZ/ParticleSeg3D/raw/main/dkfz_logo.png" height="100px" />

ParticleSeg3D is developed and maintained by the Applied Computer Vision Lab (ACVL) of [Helmholtz Imaging](http://helmholtz-imaging.de) 
and the [Division of Medical Image Computing](https://www.dkfz.de/en/mic/index.php) at the 
[German Cancer Research Center (DKFZ)](https://www.dkfz.de/en/index.html).
