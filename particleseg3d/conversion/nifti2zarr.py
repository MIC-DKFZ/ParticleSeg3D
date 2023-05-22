import argparse
from particleseg3d.utils import utils
from tqdm import tqdm
from os.path import join
import zarr
import os


def all_nifti2zarr(load_dir: str, save_dir: str) -> None:
    """
    Converts all nifti files into zarr files.

    Args:
        load_dir (str): Directory where the nifti files are located.
        save_dir (str): Directory where the zarr files should be saved.

    Returns:
        None
    """
    names = utils.load_filepaths(load_dir, extension=".nii.gz", return_path=False, return_extension=False)
    for name in tqdm(names, desc="Image conversion"):
        nifti2zarr(join(load_dir, name + ".nii.gz"), join(save_dir, name + ".zarr"))


def nifti2zarr(load_filepath: str, save_filepath: str) -> None:
    """
    Converts a single nifti file to a zarr file.

    Args:
        load_filepath (str): Path to the nifti file.
        save_dir (str): Path to where the zarr file should be saved.

    Returns:
        None
    """
    image = utils.load_nifti(load_filepath)
    image_zarr = zarr.open(save_filepath, shape=image.shape, mode='w')
    image_zarr[...] = image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the file(s) that should be converted to from Nifti to Zarr.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder that should be used for the Zarr images.")
    args = parser.parse_args()

    if not args.input.endswith(".nii.gz"):
        all_nifti2zarr(args.input, args.output)
    else:
        nifti2zarr(args.input, args.output)
