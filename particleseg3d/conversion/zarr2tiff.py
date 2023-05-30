import argparse
import os
from particleseg3d.utils import utils
import tifffile
from tqdm import tqdm
from os.path import join, basename
import zarr
from pathlib import Path


def zarr2tiff(load_dir: str, save_dir: str) -> None:
    """
    Converts a Zarr image into TIFF image slices.

    Args:
        load_dir (str): Path to the Zarr image folder.
        save_dir (str): Path to where the TIFF image slices should be saved.
    """
    name = basename(load_dir)[:-5]
    Path(join(save_dir, name)).mkdir(parents=True, exist_ok=True)
    image_zarr = zarr.open(load_dir, mode='r')
    num_slices = image_zarr.shape[0]
    for i in tqdm(range(num_slices)):
        tifffile.imwrite(join(save_dir, name, "{}.tiff".format(i)), image_zarr[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the folder that contains the Zarr image that should be converted to TIFF image slices.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder that should be used to save the TIFF image slices.")
    args = parser.parse_args()

    zarr2tiff(args.input, args.output)


if __name__ == '__main__':
    main()
