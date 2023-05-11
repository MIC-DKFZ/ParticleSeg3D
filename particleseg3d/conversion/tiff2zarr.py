import argparse
from particleseg3d.utils import utils
import tifffile
from tqdm import tqdm
from os.path import join
import zarr


def all_tiff2zarr(load_dir: str, save_dir: str) -> None:
    """
    Converts all tiff files into zarr files.

    Args:
        load_dir (str): Directory where the tiff files are located.
        save_dir (str): Directory where the zarr files should be saved.

    Returns:
        None
    """
    names = utils.load_filepaths(load_dir, return_path=False, return_extension=False)
    for name in tqdm(names, desc="Image conversion"):
        tiff2zarr(join(load_dir, name), join(save_dir, name + ".zarr"))


def tiff2zarr(load_dir: str, save_filepath: str) -> None:
    """
    Converts a set of tiff files to a zarr file.

    Args:
        load_dir (str): Path to the tiff files.
        save_dir (str): Path to where the zarr file should be saved.

    Returns:
        None
    """
    filepaths = utils.load_filepaths(join(load_dir, name), extension=["tif", "tiff", "TIF", "TIFF"])
    image = tifffile.imread(filepaths)
    image_zarr = zarr.open(save_filepath, shape=image.shape, mode='w')
    image_zarr[...] = image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the file(s) that should be converted to from Nifti to Zarr.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder that should be used for the Zarr images.")
    args = parser.parse_args()

    if not any(args.input.endswith(string) for string in ["tif", "tiff", "TIF", "TIFF"]):
        all_tiff2zarr(args.input, args.output)
    else:
        names = utils.load_filepaths(args.input, return_path=False, return_extension=False)
        for name in tqdm(names, desc="Image conversion"):
            tiff2zarr(join(args.input, name), join(args.output, name + ".zarr"))
