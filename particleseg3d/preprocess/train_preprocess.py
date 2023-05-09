import sys
from pathlib import Path
sys.path.append(str(Path('').absolute().parent))

import argparse
from particleseg3d.utils import utils
from preprocess import preprocess_all, pixel2mm
from os.path import join


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the base folder that contains the image and border-semantic folder.")
    parser.add_argument('-m', "--metadata", required=True,
                        help="Absolute path to the metadata.json.")
    parser.add_argument('-z', "--zscore", required=True,
                        help="Absolute path to the zscore_train.json.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the save folder.")
    parser.add_argument('-n', "--name", required=False, type=str, default=None, nargs="+", help="The name(s) without extension of the image(s) that should be used for training. Multiple names must be separated by spaces.")
    parser.add_argument('-t', '--task', required=True, type=int, help="The task id that should be assigned to this dataset.")
    parser.add_argument('-target_particle_size', default=(100, 100, 100), required=False, type=float, nargs=3,   # AFK_M1f: 30, SPP2315_160-250: 100
                        help="The target particle size in pixels given as three numbers separate by spaces.")
    parser.add_argument('-target_patch_size', default=(512, 512, 512), required=False, type=int, nargs=3,  # 192
                        help="The target patch size in pixels given as three numbers separate by spaces.")
    parser.add_argument('-target_spacing', default=(0.1, 0.1, 0.1), required=False, type=float, nargs=3,  # 0.01
                        help="The target spacing in millimeters given as three numbers separate by spaces.")
    parser.add_argument("--disable_standardization", required=False, default=False, action="store_true",
                        help="If image standardization should be disabled.")
    parser.add_argument('-p', '--processes', required=False, default=None, type=int, help="Number of processes to use for parallel processing. None to disable multiprocessing.")
    parser.add_argument('--auto_scale', required=False, default=False, action="store_true", help="Activates auto scaling to adjust the number of patches that should be generated.")
    parser.add_argument('--multi_size', required=False, default=False, action="store_true", help="Activates multi target particle size sampling to make the model more robust against varying particle sizes.")
    parser.add_argument('--activate_compression', required=False, default=False, action="store_true", help="Activates image compression.")
    parser.add_argument('-thickness', required=False, default=3, type=int, help="Border thickness in pixel.")
    parser.add_argument('-resample_processes', required=False, default=None, type=int, help="resample_processes")
    parser.add_argument('-zscore_norm', required=False, default="global_zscore", type=str,
                        help="zscore_norm")
    args = parser.parse_args()

    names = args.name

    if names is None:
        names = utils.load_filepaths(join(args.input, "images"), extension=".nii.gz", return_path=False, return_extension=False)

    print("Names: ", names)
    print("Num names: ", len(names))
    print("target_particle_size_in_pixel: {}, target_particle_size_in_mm: {}".format(args.target_particle_size, pixel2mm(args.target_particle_size, args.target_spacing)))

    # if args.parallel > 0:
    #     global_mp_pool.init_pool(args.parallel)

    # if len(names) == 1:
    #     dataset_name = "Task{}_{}".format(str(args.task).zfill(3), names[0])
    # else:
    #     dataset_name = "Task{}_{}".format(str(args.task).zfill(3), "particle_seg")
    dataset_name = "Task{}_{}".format(str(args.task).zfill(3), "particle_seg")

    preprocess_all(args.input, args.metadata, args.zscore, names, args.output, args.target_spacing, args.target_particle_size, args.target_patch_size,
                   not args.disable_standardization, dataset_name, True, args.processes, args.auto_scale, args.multi_size, args.activate_compression, args.thickness, args.zscore_norm, args.resample_processes)

    # global_mp_pool.close_pool()
