import argparse
import os
from particleseg3d.utils import utils
import tifffile
from tqdm import tqdm
from os.path import join
import zarr


def tiff2zarr(load_dir: str, save_dir: str, fast: bool = False) -> None:
    """
    Converts a set of TIFF image slices to a Zarr image.

    Args:
        load_dir (str): Path to the TIFF files.
        save_dir (str): Path to where the Zarr image should be saved.
    """
    name = os.path.basename(os.path.normpath(load_dir))
    filepaths = utils.load_filepaths(load_dir, extension=["tif", "tiff", "TIF", "TIFF"])
    image_shape = tifffile.imread(filepaths[0]).shape
    image_shape = (len(filepaths), *image_shape)
    image_zarr = zarr.open(join(save_dir, name + ".zarr"), shape=image_shape, mode='w')
    if not fast:
        for i, filepath in enumerate(tqdm(filepaths)):
            image_slice = tifffile.imread(filepath)
            image_zarr[i] = image_slice
    else:
        image_tiff = tifffile.imread(filepaths)
        image_zarr[...] = image_tiff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the folder that contains the TIFF image slices that should be converted to a Zarr image.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder that should be used to save the Zarr image.")
    parser.add_argument('--fast', required=False, default=False, action="store_true", help="(Optional) Fast conversion. Can be memory intensive.")
    args = parser.parse_args()

    tiff2zarr(args.input, args.output, args.fast)


if __name__ == '__main__':
    main()
