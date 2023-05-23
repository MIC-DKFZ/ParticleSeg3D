import argparse
import os
from particleseg3d.utils import utils
import tifffile
from tqdm import tqdm
from os.path import join
import zarr


def tiff2zarr(load_dir: str, save_dir: str) -> None:
    """
    Converts a set of tiff files to a zarr file.

    Args:
        load_dir (str): Path to the tiff files.
        save_dir (str): Path to where the zarr file should be saved.

    Returns:
        None
    """
    name = os.path.basename(os.path.normpath(load_dir))
    filepaths = utils.load_filepaths(load_dir, extension=["tif", "tiff", "TIF", "TIFF"])
    image = tifffile.imread(filepaths)
    image_zarr = zarr.open(join(save_dir, name + ".zarr"), shape=image.shape, mode='w')
    image_zarr[...] = image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the folder that contains the TIF image slices that should be converted to Zarr.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder that should be used to save the Zarr image.")
    args = parser.parse_args()

    tiff2zarr(args.input, args.output)
