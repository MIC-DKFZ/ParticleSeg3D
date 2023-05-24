from typing import Tuple
import tifffile
import argparse
from particleseg3d.utils import utils
from os.path import join
from tqdm import tqdm


def tiff2nifti(load_dir: str, save_dir: str, spacing: Tuple[float, float, float], name: str) -> None:
    """
        Converts a set of tiff files to a single nifti file.

        Args:
            load_dir (str): Directory where the tiff files are located.
            save_dir (str): Directory where the nifti file should be saved.
            spacing (Tuple[float, float, float]): Voxel spacing in x, y, and z directions.
            name (str): Name of the nifti file.
        """
    print("Reading filenames for {}...".format(name))
    filenames = utils.load_filepaths(join(load_dir, name), extension=["tif", "tiff", "TIF", "TIFF"])
    print("Loading tiff files...")
    image = tifffile.imread(filenames)
    print("Saving 3D image...")
    utils.save_nifti(join(save_dir, "{}.nii.gz".format(name)), image, spacing=spacing)
    print("Finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the folder of tiff images that should be converted to a single nifti image.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder, where the nifti image should be saved without the filename.")
    parser.add_argument('-s', "--spacing", required=True, type=float, nargs=3,
                        help="The image spacing given as three numbers separate by spaces.")
    parser.add_argument('-n', "--name", required=False, default=None,
                        help="The name of the image.")
    args = parser.parse_args()

    input = args.input
    output = args.output
    is_seg = args.seg
    spacing = args.spacing
    name = args.name

    if args.name is None:
        names = utils.load_filepaths(input, return_path=False, return_extension=False)

        for name in tqdm(names):
            tiff2nifti(input, output, spacing, name)
    else:
        tiff2nifti(input, output, spacing, name)
