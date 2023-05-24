from pathlib import Path
import argparse
from particleseg3d.utils import utils
from tqdm import tqdm
from os.path import join
import tifffile
import os


def all_nifti2tiff(load_dir: str, save_dir: str, stack: bool, dtype: str) -> None:
    """
    Converts all nifti files in a directory to tiff files.

    Args:
        load_dir (str): Directory where the nifti files are located.
        save_dir (str): Directory where the tiff files should be saved.
        stack (bool): True if all slices of a single image should be stacked and saved as a single tiff image.
        dtype (type): Data type of the output tiff file.
    """
    names = utils.load_filepaths(load_dir, extension=".nii.gz", return_path=False, return_extension=False)
    for name in tqdm(names, desc="Image conversion"):
        nifti2tiff(join(load_dir, name + ".nii.gz"), save_dir, name, dtype, stack)


def nifti2tiff(load_filepath: str, save_dir: str, name: str, dtype: str, stack: bool) -> None:
    """
    Converts a single nifti file to a tiff file.

    Args:
        load_filepath (str): Path to the nifti file.
        save_dir (str): Directory where the tiff file should be saved.
        name (str): Name of the tiff file.
        is_seg (bool): True if the nifti file contains a segmentation.
        stack (bool): True if all slices of the image should be stacked and saved as a single tiff image.
        dtype (type): Data type of the output tiff file.
    """
    image = utils.load_nifti(load_filepath)
    image = image.astype(eval(dtype))
    if stack:
        Path(join(save_dir, name)).mkdir(parents=True, exist_ok=True)
        for i in range(image.shape[0]):
            tifffile.imwrite(join(save_dir, name, "{}.tif".format(str(i).zfill(5))), image[i], dtype=dtype)
    else:
        tifffile.imwrite(join(save_dir, name + ".tif"), image, dtype=dtype)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the file that should be converted to from Nifti to TIF.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder that should be used for the tif images.")
    parser.add_argument("--dtype", required=False, default="np.uint16", type=str, help="Data type of the output tiff file.")
    parser.add_argument("--stack", required=False, default=False, action="store_true", help="If all slices of a single image should be stacked and saved as single tif image.")
    args = parser.parse_args()

    if not args.input.endswith(".nii.gz"):
        all_nifti2tiff(args.input, args.output, args.seg, args.stack)
    else:
        save_dir = os.path.dirname(args.output)
        name = os.path.splitext(os.path.basename(args.output))[0]
        nifti2tiff(args.input, save_dir, name, args.dtype, args.stack)
