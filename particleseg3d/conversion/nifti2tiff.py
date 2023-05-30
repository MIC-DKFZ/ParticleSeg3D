import tifffile
import argparse
from particleseg3d.utils import utils
from os.path import join
from tqdm import tqdm
import os
from pathlib import Path


def nifti2tiff(load_filepath: str, save_dir: str) -> None:
    """
    Converts a NIFTI image to a set of TIFF images.

    Args:
        load_dir (str): Path to the NIFTI image.
        save_dir (str): Path to where the TIFF image slices should be saved.
    """
    name = os.path.basename(os.path.normpath(load_filepath))[:-7]
    Path(join(save_dir, name)).mkdir(parents=True, exist_ok=True)
    image = utils.load_nifti(load_filepath)
    num_slices = image.shape[0]
    for i in tqdm(range(num_slices)):
        tifffile.imwrite(join(save_dir, name, "{}.tiff".format(i)), image[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the NIFTI image that should be converted to a set of TIFF images slices.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder, where the TIFF images slices should be saved.")
    args = parser.parse_args()

    nifti2tiff(args.input, args.output)


if __name__ == '__main__':
    main()
