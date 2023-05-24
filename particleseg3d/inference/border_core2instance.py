import GeodisTK
import numpy as np
from scipy.ndimage import label as nd_label
import cc3d
from tqdm import tqdm
import copy
import zarr
from acvl_utils.miscellaneous.ptqdm import ptqdm
import numpy_indexed as npi
from skimage.morphology import cube
from skimage.morphology import dilation
from scipy.ndimage.morphology import distance_transform_edt
from typing import Tuple, Optional, Type


def border_core2instance(border_core: np.ndarray, pred_border_core_tmp_filepath: str, processes: Optional[int] = None, progressbar: bool = True, dtype: Type = np.uint16) -> Tuple[np.ndarray, int]:
    """
    Convert the border-core segmentation of an entire image into an instance segmentation.

    Args:
        border_core (np.ndarray): The border-core segmentation of the entire image.
        pred_border_core_tmp_filepath (str): Temporary file path to store the intermediate result.
        processes (Optional[int], default=None): Number of processes to use. If None, it uses a single process.
        progressbar (bool, default=True): Whether to show progress bar.
        dtype (Type, default=np.uint16): The data type for the output segmentation.

    Returns:
        Tuple[np.ndarray, int]: The instance segmentation of the entire image, Number of instances.
    """

    border_core_array = np.array(border_core)
    component_seg = cc3d.connected_components(border_core_array > 0)
    component_seg = component_seg.astype(dtype)
    instances = np.zeros_like(border_core, dtype=dtype)
    num_instances = 0
    props = {i: bbox for i, bbox in enumerate(cc3d.statistics(component_seg)["bounding_boxes"])}
    del props[0]
    component_seg = zarr.array(component_seg, chunks=(100, 100, 100))
    zarr.save(pred_border_core_tmp_filepath, component_seg)

    component_seg = zarr.open(pred_border_core_tmp_filepath, mode='r')

    border_core_component2instance = border_core_component2instance_dilation

    if processes is None or processes == 0:
        for index, (label, bbox) in enumerate(tqdm(props.items(), desc="Border-Core2Instance", disable=not progressbar)):
            filter_mask = component_seg[bbox] == label
            border_core_patch = copy.deepcopy(border_core[bbox])
            border_core_patch[filter_mask != 1] = 0
            instances_patch = border_core_component2instance(border_core_patch).astype(dtype)
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

        instances_patches = ptqdm(border_core_component2instance, border_core_patches, processes, desc="Border-Core2Instance", disable=not progressbar)

        for index, (label, bbox) in enumerate(tqdm(props.items())):
            instances_patch = instances_patches[index].astype(dtype)
            instances_patch[instances_patch > 0] += num_instances
            num_instances = max(num_instances, int(np.max(instances_patch)))
            patch_labels = np.unique(instances_patch)
            patch_labels = patch_labels[patch_labels > 0]
            for patch_label in patch_labels:
                instances[bbox][instances_patch == patch_label] = patch_label

    return instances, num_instances


def border_core_component2instance_distance(patch: np.ndarray, core_label: int = 1, border_label: int = 2) -> np.ndarray:
    """
    Convert a patch that consists of an entire connected component of the border-core segmentation into an instance segmentation using a geodesic distance-based approach.

    This method calculates the shortest path distance (geodesic distance) between each voxel and the set of target voxels (core) and assigns each voxel to the instance (core) that it is closest to.

    :param patch: An entire connected component patch of the border-core segmentation.
    :param core_label: The core label.
    :param border_label: The border label.
    :return: The instance segmentation of this connected component patch.
    """
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

    return instances


def border_core_component2instance_dilation(patch: np.ndarray, core_label: int = 1, border_label: int = 2) -> np.ndarray:
    """
    Convert a patch that consists of an entire connected component of the border-core segmentation into an instance segmentation using a morphological dilation operation.

    This method starts with the core instances and progressively dilates them until all border voxels are covered, hence performing the instance segmentation.

    :param patch: An entire connected component patch of the border-core segmentation.
    :param core_label: The core label.
    :param border_label: The border label.
    :return: The instance segmentation of this connected component patch.
    """
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label, output=core_instances)
    if num_instances == 0:
        return patch
    patch, core_instances, num_instances = remove_small_cores(patch, core_instances, core_label, border_label)
    core_instances = np.zeros_like(patch, dtype=np.uint16)
    num_instances = nd_label(patch == core_label, output=core_instances)  # remove_small_cores invalidates the previous core_instances, so recompute it. The computation time is neglectable.
    if num_instances == 0:
        return patch
    instances = copy.deepcopy(core_instances)
    border = patch == border_label
    while np.sum(border) > 0:
        ball_here = cube(3)

        dilated = dilation(core_instances, ball_here)
        dilated[patch == 0] = 0
        diff = (core_instances == 0) & (dilated != core_instances)
        instances[diff & border] = dilated[diff & border]
        border[diff] = 0
        core_instances = dilated

    return instances


def remove_small_cores(
    patch: np.ndarray, 
    core_instances: np.ndarray, 
    core_label: int, 
    border_label: int, 
    min_distance: float = 1, 
    min_ratio_threshold: float = 0.95, 
    max_distance: float = 3, 
    max_ratio_threshold: float = 0.0
) -> Tuple[np.ndarray, np.ndarray, int]: 
    """
    Removes small cores in a patch based on the distance transform of the core label.

    Args:
        patch (np.ndarray): An entire connected component patch of the border-core segmentation.
        core_instances (np.ndarray): The labeled core instances in the patch.
        core_label (int): The label for cores.
        border_label (int): The label for borders.
        min_distance (float, default=1): The minimum distance for removal.
        min_ratio_threshold (float, default=0.95): The minimum ratio threshold for removal.
        max_distance (float, default=3): The maximum distance for removal.
        max_ratio_threshold (float, default=0.0): The maximum ratio threshold for removal.

    Returns:
        Tuple[np.ndarray, np.ndarray, int]: The updated patch after removing small cores, the updated core instances, and the number of cores.
    """

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
        if (min_ratio_threshold is None or min_ratio >= min_ratio_threshold) and (max_ratio_threshold is None or max_ratio <= max_ratio_threshold):
            core_ids_to_remove.append(core_id)

    num_cores = len(core_ids) - len(core_ids_to_remove)

    if len(core_ids_to_remove) > 0:
        target_values = np.zeros_like(core_ids_to_remove, dtype=int)
        shape = patch.shape
        core_instances = npi.remap(core_instances.flatten(), core_ids_to_remove, target_values)
        core_instances = core_instances.reshape(shape)

        patch[(patch == core_label) & (core_instances == 0)] = border_label

    return patch, core_instances, num_cores

