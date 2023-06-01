import argparse
from particleseg3d.utils import utils
import zarr


def nifti2zarr(load_filepath: str, save_filepath: str) -> None:
    """
    Converts a single nifti file to a zarr file.

    Args:
        load_filepath (str): Path to the nifti file.
        save_dir (str): Path to where the zarr file should be saved.
    """
    image = utils.load_nifti(load_filepath)
    image_zarr = zarr.open(save_filepath, shape=image.shape, mode='w')
    image_zarr[...] = image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the NIFTI image that should be converted to from Nifti to Zarr.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the file that should be used for the Zarr image.")
    args = parser.parse_args()

    nifti2zarr(args.input, args.output)


if __name__ == '__main__':
    main()
