from particleseg3d.utils import utils
import argparse
from tqdm import tqdm
from os.path import join


def all_change_spacing(load_filepath, save_filepath, new_spacing, micron2mm):
    names = utils.load_filepaths(load_filepath, return_path=False, return_extension=False)
    for name in tqdm(names, desc="Spacing conversion"):
        change_spacing(join(load_filepath, name + ".nii.gz"), join(save_filepath, name + ".nii.gz"), new_spacing, micron2mm)


def change_spacing(load_filepath, save_filepath, new_spacing, micron2mm):
    image, spacing, affine, header = utils.load_nifti(load_filepath, return_meta=True)

    if new_spacing is None:
        if micron2mm:
            new_spacing = [s / 1000 for s in spacing]
        else:
            raise RuntimeError("Either a spacing must be defined or micron2mm must be activated.")

    utils.save_nifti(save_filepath, image, spacing=new_spacing)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=False,
                        help="Absolute input path to the folder or file that should be converted to border-semantic segmentation. In case a folder is given, all .nii.gz files will be converted.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder or file that should be used for saving the border-semantic segmentations.")
    parser.add_argument('-s', "--spacing", required=False, type=float, nargs=3,
                        help="The image spacing given as three numbers separate by spaces.")
    parser.add_argument('--micron2mm', required=False, default=False, action="store_true", help="Convert micron spacing to mm spacing.")
    args = parser.parse_args()

    input = args.input
    output = args.output
    spacing = args.spacing
    micron2mm = args.micron2mm

    if not args.input.endswith(".nii.gz"):
        all_change_spacing(input, output, spacing, micron2mm)
    else:
        change_spacing(input, output, spacing, micron2mm)
