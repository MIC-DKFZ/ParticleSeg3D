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
To install the ParticleSeg3D project, you'll first need to clone the repository to your local machine.

1. Open your terminal and clone the repository:
    ```cmd
    git clone https://github.com/MIC-DKFZ/ParticleSeg3D.git
    ```
2. Navigate into the cloned repository:
    ```cmd
    cd ParticleSeg3D
    ```
3. Install the project using pip:
    ```cmd
    pip install .
    ```
You should now have the ParticleSeg3D package installed in your Python environment, and you'll be able to use all ParticleSeg3D commands from anywhere on your system.

## Usage - Inference

### Model download
ParticleSeg3D requires a trained model in order to run inference. The trained model can be downloaded [here](https://syncandshare.desy.de/index.php/s/id9D9pkATrFw65s). After downloading the weights, the weights need to be unpacked and saved at a location of your choosing.

### Conversion to Zarr
To run inference on an image using ParticleSeg3D, the image must first be converted into the Zarr format. In case of a series of TIFF image files, this conversion can be accomplished using the following command from anywhere on the system:
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
You can run inference on Zarr images from anywhere on the system using the ps3d_inference command. The Zarr images need to be located in a folder named 'images' and the 'metadata.json' needs to be placed at it parent folder such that the folder structure looks like this:
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
...