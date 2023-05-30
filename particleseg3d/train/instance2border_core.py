import numpy as np
from skimage.morphology import ball
from skimage.morphology import binary_erosion
from skimage.measure import regionprops
from typing import Dict, Any


def instance2border_core(instance_seg: np.ndarray, thickness: int, border_label: int = 2, core_label: int = 1) -> np.ndarray:
    """
    Converts a label instance segmentation into a binary segmentation with border and core labels.

    Args:
        instance_seg (np.ndarray): Instance segmentation labels.
        thickness (int): The thickness of the border.
        border_label (int, optional): The label value for the border. Defaults to 2.
        core_label (int, optional): The label value for the core. Defaults to 1.

    Returns:
        np.ndarray: The converted binary segmentation labels.
    """
    instance_seg = instance_seg.astype(np.uint16)
    selem = ball(thickness, dtype=int)
    border_semantic = np.zeros_like(instance_seg, dtype=np.uint8)

    for instance in regionprops(instance_seg):
        border_semantic_particle = instance2border_core_particle(instance, instance_seg, selem, border_label, core_label, thickness + 3)
        i_start, j_start, k_start, i_end, j_end, k_end = border_semantic_particle["bbox"]
        mask = border_semantic_particle["image"]
        border_semantic[i_start:i_end, j_start:j_end, k_start:k_end][mask == border_label] = border_label
        border_semantic[i_start:i_end, j_start:j_end, k_start:k_end][mask == core_label] = core_label

    return border_semantic


def instance2border_core_particle(instance: Dict[str, Any], instance_seg: np.ndarray, selem: np.ndarray, border_label: int, core_label: int, roi_padding: int) -> Dict[str, Any]:
    """
    Converts a label instance segmentation of a particle into a binary segmentation with border and core labels.

    Args:
        instance (Dict[str, Any]): Dictionary of the particle instance properties.
        instance_seg (np.ndarray): Instance segmentation labels.
        selem (np.ndarray): Structuring element for erosion.
        border_label (int): The label value for the border.
        core_label (int): The label value for the core.
        roi_padding (int): Padding for the region of interest.

    Returns:
        Dict[str, Any]: The converted binary segmentation labels with the region of interest.
    """
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
