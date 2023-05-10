import sys
from pathlib import Path
sys.path.append(str(Path('').absolute().parent))

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join
from skimage.morphology import ball
from skimage.morphology import binary_erosion
from skimage.measure import regionprops
import os
from particleseg3d.utils import utils, global_mp_pool
import argparse
from tqdm import tqdm
from functools import partial


def all_instance2border_core(load_filepath, save_filepath, border_label=2, core_label=1, border_thickness_in_pixel=5, parallel=0):
    filenames = utils.load_filepaths(load_filepath)
    for filename in tqdm(filenames, desc="Image conversion"):
        single_instance2border_core(filename, join(save_filepath, os.path.basename((filename))), border_label, core_label, border_thickness_in_pixel, parallel)


def single_instance2border_core(load_filepath, save_filepath, border_label=2, core_label=1, border_thickness_in_pixel=5, parallel=0):
    label_img, spacing, affine, header = utils.load_nifti(load_filepath, is_seg=True, return_meta=True)
    border_semantic = instance2border_core_process(label_img, border_label=border_label, core_label=core_label, border_thickness_in_pixel=border_thickness_in_pixel, progress_bar=True, parallel=parallel)
    utils.save_nifti(save_filepath, border_semantic, is_seg=True, spacing=spacing, dtype=np.uint16)


# Works only for isotropic images. generate_ball uses size_conversion_factor, which is a single value -> Same thickness in every dimension.
def instance2border_core_process(instance_seg, border_label=2, core_label=1, border_thickness_in_pixel=5, progress_bar=False, parallel=0):
    # start_time = time.time()
    instance_seg = instance_seg.astype(np.uint16)
    selem = ball(border_thickness_in_pixel, dtype=int)
    border_semantic_particles = []

    if parallel == 0:
        for instance in tqdm(regionprops(instance_seg), desc="Instance conversion", disable=not progress_bar):
            border_semantic_particle = instance2border_core_particle(instance, instance_seg, selem, border_label, core_label, border_thickness_in_pixel + 3)
            border_semantic_particles.append(border_semantic_particle)
    else:  # parallel is for some reason much slower. Don't use!
        pool, _ = global_mp_pool.get_pool()
        instances = []
        for instance in regionprops(instance_seg):
            instances.append({"bbox": instance["bbox"], "label": instance["label"]})
        pool.map(partial(instance2border_core_particle, instance_seg=instance_seg, selem=selem, border_label=border_label, core_label=core_label, roi_padding=border_thickness_in_pixel + 3), instances)

    border_semantic = np.zeros_like(instance_seg, dtype=np.uint8)
    for border_semantic_particle in border_semantic_particles:
        i_start, j_start, k_start, i_end, j_end, k_end = border_semantic_particle["bbox"]
        mask = border_semantic_particle["image"]
        border_semantic[i_start:i_end, j_start:j_end, k_start:k_end][mask == border_label] = border_label
        border_semantic[i_start:i_end, j_start:j_end, k_start:k_end][mask == core_label] = core_label

    # print("Elapsed time: ", time.time() - start_time)
    return border_semantic


def instance2border_core_particle(instance, instance_seg, selem, border_label, core_label, roi_padding):
    border_semantic_particle = {}
    i_start, j_start, k_start, i_end, j_end, k_end = instance["bbox"]
    # Pad the roi to improve quality of the erosion
    i_start, j_start, k_start, i_end, j_end, k_end = max(0, i_start - roi_padding), max(0, j_start - roi_padding), max(0, k_start - roi_padding), \
                                                     min(instance_seg.shape[0], i_end + roi_padding), min(instance_seg.shape[1], j_end + roi_padding), min(instance_seg.shape[2], k_end + roi_padding)
    border_semantic_particle["bbox"] = i_start, j_start, k_start, i_end, j_end, k_end
    roi_mask = instance_seg[i_start:i_end, j_start:j_end, k_start:k_end] == instance["label"]
    border_semantic_particle["image"] = roi_mask.astype(np.uint8)
    eroded = binary_erosion(roi_mask, selem)
    border_semantic_particle["image"][(eroded == 0) & (roi_mask == 1)] = border_label
    border_semantic_particle["image"][(eroded == 1) & (roi_mask == 1)] = core_label
    return border_semantic_particle


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the folder or file that should be converted to border-semantic segmentation. In case a folder is given, all .nii.gz files will be converted.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the folder or file that should be used for saving the border-semantic segmentations.")
    parser.add_argument('-b', '--border_thickness', required=False, default=5, type=int, help="The border-thickness in pixel.")
    parser.add_argument('-p', '--parallel', required=False, default=0, type=int, help="Number of threads to use for parallel processing. 0 to disable multiprocessing.")
    args = parser.parse_args()

    input = args.input
    output = args.output

    if args.parallel > 0:
        global_mp_pool.init_pool(args.parallel)

    if input.endswith(".nii.gz"):
        single_instance2border_core(input, output, parallel=args.parallel, border_thickness_in_pixel=args.border_thickness)
    else:
        all_instance2border_core(input, output, parallel=args.parallel, border_thickness_in_pixel=args.border_thickness)

    global_mp_pool.close_pool()
