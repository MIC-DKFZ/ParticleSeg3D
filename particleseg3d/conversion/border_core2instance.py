import sys
from pathlib import Path
sys.path.append(str(Path('').absolute().parent))

import GeodisTK
import numpy as np
from scipy.ndimage import label as nd_label
import cc3d
from tqdm import tqdm
import copy
from particleseg3d.utils import utils
from particleseg3d.utils.imap_tqdm import imap_tqdm
import numpy_indexed as npi
from skimage.morphology import cube
from skimage.morphology import dilation
from scipy.ndimage.morphology import distance_transform_edt


def border_core2instance(border_core, processes=None, progressbar=True, dtype=np.uint16):
    """
    Convert the border-core segmentation of an entire image into an instance segmentation.
    :param border_core: The border-core segmentation of the entire image.
    :param progressbar: If a progressbar should be shown.
    :return: The instance segmentation of the entire image.
    """
    border_core_array = np.array(border_core)
    component_seg = cc3d.connected_components(border_core_array > 0)
    component_seg = component_seg.astype(dtype)
    instances = np.zeros_like(border_core, dtype=dtype)
    num_instances = 0
    props = {i: bbox for i, bbox in enumerate(cc3d.statistics(component_seg)["bounding_boxes"])}
    border_core_component2instance = border_core_component2instance_dilation

    if processes is None:
        for index, (label, bbox) in enumerate(tqdm(props.items(), desc="Border-Core2Instance", disable=not progressbar)):
            filter_mask = component_seg[bbox] == label
            border_core_patch = copy.deepcopy(border_core[bbox])
            border_core_patch[filter_mask != 1] = 0
            instances_patch = border_core_component2instance(border_core_patch, dtype=dtype).astype(dtype)
            instances_patch[instances_patch > 0] += num_instances
            num_instances = max(num_instances, np.max(instances_patch))
            patch_labels = np.unique(instances_patch)
            patch_labels = patch_labels[patch_labels > 0]
            for patch_label in patch_labels:
                instances[bbox][instances_patch == patch_label] = patch_label
    else:
        border_core_patches = []
        for index, (label, bbox) in enumerate(props.items()):
            filter_mask = component_seg[bbox] == label
            border_core_patch = copy.deepcopy(border_core[bbox])
            border_core_patch[filter_mask != 1] = 0
            border_core_patches.append(border_core_patch)

        instances_patches = imap_tqdm(border_core_component2instance, border_core_patches, processes, desc="Border-Core2Instance", disable=not progressbar, dtype=dtype)

        for index, (label, bbox) in enumerate(tqdm(props.items())):
            instances_patch = instances_patches[index].astype(dtype)
            instances_patch[instances_patch > 0] += num_instances
            num_instances = max(num_instances, int(np.max(instances_patch)))
            patch_labels = np.unique(instances_patch)
            patch_labels = patch_labels[patch_labels > 0]
            for patch_label in patch_labels:
                instances[bbox][instances_patch == patch_label] = patch_label

    # print("instances mislabeled? ", is_mislabeled(instances))
    # print(instances.max())
    # print(len(np.unique(instances)))

    return instances, num_instances


def border_core_component2instance_distance(patch, core_label=1, border_label=2, dtype=np.uint16):
    """
    Convert a patch that consists of an entire connected component of the border-core segmentation into an instance segmentation.
    :param patch: An entire connected component patch of the border-core segmentation.
    :param core_label: The core label.
    :param border_label: The border label.
    :return: The instance segmentation of this connected component patch.
    """
    # save_dir = "/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/tmp"
    # name = str(uuid.uuid4())[:8]
    # utils.save_nifti(join(save_dir, "{}_border_core_before.nii.gz".format(name)), patch)
    # core_instances = cc3d.connected_components(patch == core_label)
    # core_instances = core_instances.astype(dtype)
    # original_patch = copy.deepcopy(patch)
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label, output=core_instances)
    # original_core_instances = copy.deepcopy(core_instances)
    if num_instances == 0:
        return patch
    patch, core_instances, num_instances = remove_small_cores(patch, core_instances, core_label, border_label)
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label, output=core_instances)  # remove_small_cores invalidates the previous core_instances, so recompute it. The computation time is neglectable.
    if num_instances == 0:
        return patch

    foreground = (patch > 0).astype(np.uint8)
    foreground[foreground == 0] = 255
    foreground[foreground == 1] = 0
    background = 255 - foreground

    distances = []
    distances.append(background)
    for core_label in range(1, num_instances+1):
        core = core_instances == core_label
        core_distances = GeodisTK.geodesic3d_raster_scan(foreground, core, (1, 1, 1), 1, 1)
        distances.append(core_distances)
    distances = np.asarray(distances)

    instances = np.argmin(distances, axis=0)

    # mislabled, counts = is_mislabeled(instances)
    # if mislabled:
    #     for i, distance in enumerate(distances):
    #         utils.save_nifti("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/quantification/tmp/tmp_distance_{}.nii.gz".format(i), distance)
    #     utils.save_nifti("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/quantification/tmp/tmp_border_core.nii.gz", original_patch)
    #     utils.save_nifti("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/quantification/tmp/tmp_core_instances_original.nii.gz", original_core_instances)
    #     utils.save_nifti("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/quantification/tmp/tmp_core_instances.nii.gz", core_instances)
    #     utils.save_nifti("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/quantification/tmp/tmp_instance.nii.gz", instances)
    #     print("counts: {}".format(counts))

    return instances


def border_core_component2instance_dilation(patch, core_label=1, border_label=2, dtype=np.uint16):
    # spacing = np.array(spacing)
    # utils.save_nifti("/home/k539i/Documents/network_drives/cluster-data/original/2021_Gotkowski_HZDR-HIF/evaluation/predictions/Task313_nnUNetTrainerV2/predictions/p_0.005_rc_None_ac_None/GR_AR1_16_012_patch_3/tmp1.nii.gz", patch)
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label, output=core_instances)
    if num_instances == 0:
        return patch
    # patch, core_instances, num_instances = remove_small_cores(patch, core_instances, core_label, border_label, min_rel_core_size, min_abs_core_size)
    patch, core_instances, num_instances = remove_small_cores(patch, core_instances, core_label, border_label)
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label, output=core_instances)  # remove_small_cores invalidates the previous core_instances, so recompute it. The computation time is neglectable.
    if num_instances == 0:
        return patch
    instances = copy.deepcopy(core_instances)
    border = patch == border_label

    # utils.save_nifti("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/quantification/tmp/tmp_border_core.nii.gz", patch)

    # already_dilated_mm = np.array((0, 0, 0))
    while np.sum(border) > 0:
        # print("tmp")
        # strel_size = [0, 0, 0]
        # maximum_dilation = max(already_dilated_mm)
        # for i in range(3):
        #     if spacing[i] == min(spacing):
        #         strel_size[i] = 1
        #         continue
        #     if already_dilated_mm[i] + spacing[i] / 2 < maximum_dilation:
        #         strel_size[i] = 1
        # ball_here = ball(1)
        #
        # if strel_size[0] == 0: ball_here = ball_here[1:2]
        # if strel_size[1] == 0: ball_here = ball_here[:, 1:2]
        # if strel_size[2] == 0: ball_here = ball_here[:, :, 1:2]
        ball_here = cube(3)

        dilated = dilation(core_instances, ball_here)
        # utils.save_nifti("/home/k539i/Documents/network_drives/cluster-data/original/2021_Gotkowski_HZDR-HIF/evaluation/predictions/Task313_nnUNetTrainerV2/predictions/p_0.005_rc_None_ac_None/GR_AR1_16_012_patch_3/tmp2.nii.gz", dilated)
        dilated[patch == 0] = 0
        # utils.save_nifti("/home/k539i/Documents/network_drives/cluster-data/original/2021_Gotkowski_HZDR-HIF/evaluation/predictions/Task313_nnUNetTrainerV2/predictions/p_0.005_rc_None_ac_None/GR_AR1_16_012_patch_3/tmp3.nii.gz", dilated)
        diff = (core_instances == 0) & (dilated != core_instances)
        instances[diff & border] = dilated[diff & border]
        border[diff] = 0
        core_instances = dilated
        # already_dilated_mm = [already_dilated_mm[i] + spacing[i] if strel_size[i] == core_label else already_dilated_mm[i] for i in range(3)]

    # utils.save_nifti("/home/k539i/Documents/datasets/original/2021_Gotkowski_HZDR-HIF/quantification/tmp/tmp_instance.nii.gz", instances)

    return instances


# def remove_small_cores(patch, core_instances, core_label, border_label, min_rel_core_size, min_abs_core_size=15):  # factor = 0.00013888888 -> pixels = 30
#     """
#     Very small core regions can exist due to miss-predictions, which can result in interpreting a single instance as multiple. This should be avoided.
#     Therefore, we convert cores that are too small into border.
#     """
#     if min_rel_core_size is None:
#         min_rel_core_size = 0.0
#     if min_abs_core_size is None:
#         min_abs_core_size = 0.0
#     # min_abs_core_size = np.rint(np.power(target_particle_size, 3) * min_num_core_factor).astype(int)
#     core_ids, core_sizes = np.unique(core_instances, return_counts=True)
#
#     core_sizes = core_sizes[core_ids > 0]
#     core_ids = core_ids[core_ids > 0]
#     max_core_size = np.max(core_sizes)
#     min_rel_core_size = max_core_size * min_rel_core_size
#     # print("min_abs_core_size: {}, min_rel_threshold: {}, target_particle_size: {}".format(min_abs_core_size, min_rel_threshold, target_particle_size))
#     min_core_size = min_rel_core_size if min_rel_core_size > min_abs_core_size else min_abs_core_size
#     core_ids_to_remove = core_ids[core_sizes < min_core_size]
#     num_cores = len(core_ids) - len(core_ids_to_remove)
#
#     if len(core_ids_to_remove) > 0:
#         target_values = np.zeros_like(core_ids_to_remove, dtype=int)
#         shape = patch.shape
#         core_instances = npi.remap(core_instances.flatten(), core_ids_to_remove, target_values)
#         core_instances = core_instances.reshape(shape)
#
#         patch[(patch == core_label) & (core_instances == 0)] = border_label
#
#     return patch, core_instances, num_cores


def remove_small_cores(patch, core_instances, core_label, border_label, min_distance=1, min_ratio_threshold=0.95, max_distance=3, max_ratio_threshold=0.0):  # 0.95,  0.05
    distances = distance_transform_edt(patch == core_label)
    core_ids = np.unique(core_instances)

    core_ids_to_remove = []
    for core_id in core_ids:
        core_distances = distances[core_instances == core_id]
        num_min_distances = np.count_nonzero(core_distances <= min_distance)
        num_max_distances = np.count_nonzero(core_distances >= max_distance)
        num_core_voxels = np.count_nonzero(core_instances == core_id)
        min_ratio = num_min_distances / num_core_voxels
        max_ratio = num_max_distances / num_core_voxels
        # if num_below_threshold / num_core_voxels >= threshold:
        if (min_ratio_threshold is None or min_ratio >= min_ratio_threshold) and (max_ratio_threshold is None or max_ratio <= max_ratio_threshold):
        # if num_below_threshold > 0:
            core_ids_to_remove.append(core_id)

    num_cores = len(core_ids) - len(core_ids_to_remove)
    # print("num removed cores: ", len(core_ids_to_remove))

    if len(core_ids_to_remove) > 0:
        target_values = np.zeros_like(core_ids_to_remove, dtype=int)
        shape = patch.shape
        core_instances = npi.remap(core_instances.flatten(), core_ids_to_remove, target_values)
        core_instances = core_instances.reshape(shape)

        patch[(patch == core_label) & (core_instances == 0)] = border_label

    return patch, core_instances, num_cores


if __name__ == '__main__':
    load_filepath = "/home/k539i/Documents/network_drives/cluster-data/original/2021_Gotkowski_HZDR-HIF/evaluation/predictions/Task313_nnUNetTrainerV2/predictions/p_0.005_rc_None_ac_None/AFK_M4c_patch_1/AFK_M4c_patch_1_border_core_zoomed.nii.gz"
    save_filepath = "/home/k539i/Documents/network_drives/cluster-data/original/2021_Gotkowski_HZDR-HIF/evaluation/predictions/Task313_nnUNetTrainerV2/predictions/p_0.005_rc_None_ac_None/AFK_M4c_patch_1/AFK_M4c_patch_1_border_tmp.zarr"

    border_core = utils.load_nifti(load_filepath)

    border_core2instance(border_core, save_filepath, processes=24, reuse=True)
