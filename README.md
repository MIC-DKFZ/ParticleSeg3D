# ParticleSeg3D: Scalable, out-of-the-box segmentation of individual particles from mineral samples acquired with micro CT

This repository contains the code implementation for the paper titled "Scalable, out-of-the-box segmentation of individual particles from mineral samples acquired with micro CT." The paper presents a method for extracting individual particles from large micro CT images taken from mineral samples embedded in an epoxy matrix.

## Abstract

Minerals are indispensable for a functioning modern society. However, their supply is limited, highlighting the need for optimizing their exploration and extraction. Traditional approaches for particle analysis rely on bulk segmentation and characterization of particles, often requiring manual separation of touching particles and specific retraining for each new image. This paper proposes an instance segmentation method that addresses these limitations and enables scalable experiments. The method is based on the nnU-Net framework and utilizes a particle size normalization and border-core representation. It is trained on a large dataset containing particles of various materials and minerals, allowing it to be applied out-of-the-box to a wide range of particle types without the need for further manual annotations or retraining.

## Usage

To use the code, follow these steps:

1. Install the required dependencies and packages mentioned in the `requirements.txt` file.

2. Prepare your input dataset in the required format, including the micro CT images, metadata file, and zscore file.

3. Set the necessary configuration parameters in the code, such as the input and output paths, model directory, and other parameters.

4. Run the script `main.py` with appropriate command-line arguments to perform the prediction and segmentation.

Please refer to the paper and the provided code for more details on the methodology and usage instructions.

## Code Overview

The code provided in this repository implements the functionality described in the paper. Here's an overview of the main components and functions:

```python
from torch.utils.data import DataLoader
from particleseg3d.utils import utils
import pytorch_lightning as pl
from os.path import join
# ... (continue with the rest of the code snippet)
