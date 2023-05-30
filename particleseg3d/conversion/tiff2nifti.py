from typing import Tuple
import tifffile
import argparse
from particleseg3d.utils import utils
from os.path import join
from tqdm import tqdm
import os


def tiff2nifti(load_dir: str, save_dir: str, spacing: Tuple[float, float, float]) -> None:
    """
    Converts a set of TIFF image slices to a NIFTI image.

    Args:
        load_dir (str): Path to the TIFF files.
        save_dir (str): Path to where the NIFTI image should be saved.
    """
    name = os.path.basename(os.path.normpath(load_dir))
    filepaths = utils.load_filepaths(load_dir, extension=["tif", "tiff", "TIF", "TIFF"])
    image = tifffile.imread(filepaths)
    utils.save_nifti(join(save_dir, name + ".nii.gz"), image, spacing=spacing)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the folder of TIFF images that should be converted to a single NIFTI image.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder, where the NIFTI image should be saved without the filename.")
    parser.add_argument('-s', "--spacing", required=True, type=float, nargs=3,
                        help="The image spacing given as three numbers separate by spaces.")
    args = parser.parse_args()

    tiff2nifti(args.input, args.output, args.spacing)


if __name__ == '__main__':
    main()
