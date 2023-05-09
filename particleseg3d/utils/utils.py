try:
    import global_mp_pool
except ModuleNotFoundError as e:
    import sys
    from pathlib import Path
    sys.path.append(str(Path('').absolute().parent))
    from particle_seg.helper import global_mp_pool

import numpy as np
import SimpleITK as sitk
import torch
from torch.nn import functional
import os
from natsort import natsorted
from os.path import join
import torchio as tio
from skimage.transform import resize
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial


def resample(image: np.ndarray, target_shape, is_seg=False) -> np.ndarray:
    if all([i == j for i, j in zip(image.shape, target_shape)]):
        return image

    with torch.no_grad():
        image = torch.from_numpy(image.astype(np.float32)).cuda()
        if not is_seg:
            image = functional.interpolate(image[None, None], target_shape, mode='trilinear')[0, 0]
        else:
            image = functional.interpolate(image[None, None], target_shape, mode='nearest')[0, 0]
        image = image.cpu().numpy()
    torch.cuda.empty_cache()
    return image


def resample_tio(subject, target_shape, gpu=True, processes=None):
    image = subject["image"].tensor
    # _target_shape = np.asarray(target_shape)[np.newaxis, ...]
    if all([i == j for i, j in zip(image.shape[1:], target_shape)]):
        return subject

    with torch.no_grad():
        image = image.float()
        if gpu:
            image = image.cuda()
        image = functional.interpolate(image[np.newaxis, ...], target_shape, mode='trilinear')[0]
    subject["image"] = tio.ScalarImage(tensor=image.cpu())

    if "label" in subject:
        seg = subject["label"].tensor
        with torch.no_grad():
            seg = seg.float()
            if gpu:
                seg = seg.cuda()
            # seg = functional.interpolate(seg[np.newaxis, ...], target_shape, mode='nearest')[0]
            if processes is None:
                seg = resize_segmentation_tensor(seg[np.newaxis, ...], target_shape)[0]
            else:
                seg = resize_segmentation_tensor_parallel(seg[np.newaxis, ...], target_shape, processes)[0]
        subject["label"] = tio.LabelMap(tensor=seg.cpu())

    torch.cuda.empty_cache()
    return subject


# def resize_segmentation_np(seg: np.ndarray, target_shape, order=1) -> np.ndarray:
#     reshaped = np.zeros(target_shape, dtype=seg.dtype)
#     unique_labels = np.unique(seg)
#
#     for i, label in enumerate(unique_labels):
#         mask = seg == label
#         reshaped_multihot = resize(mask.astype(float), target_shape, order, mode="edge", clip=True, anti_aliasing=False)
#         reshaped[reshaped_multihot >= 0.5] = label
#     return reshaped


def smooth_seg_resize_np(seg: np.ndarray, target_shape, order=1, labels=None, continuous=True) -> np.ndarray:
    """Order should be between 1-3. The higher the smoother, but also longer."""
    reshaped = np.zeros(target_shape, dtype=seg.dtype)
    if labels is None:
        if continuous:
            labels = list(range(np.max(seg) + 1))
        else:
            labels = np.unique(seg)

    for i, label in enumerate(labels):
        mask = seg == label
        reshaped_multihot = resize(mask.astype(float), target_shape, order, mode="edge", clip=True, anti_aliasing=False)
        reshaped[reshaped_multihot >= 0.5] = label
    return reshaped


def resize_segmentation_tensor(seg: torch.Tensor, target_shape) -> torch.Tensor:
    target_shape_tmp = np.asarray(seg.shape)
    target_shape_tmp[2:] = np.asarray(target_shape)
    target_shape_tmp = tuple(target_shape_tmp)
    reshaped = torch.zeros(target_shape_tmp, dtype=seg.dtype, device=seg.device)
    unique_labels = torch.unique(seg, sorted=True)

    for i, label in enumerate(tqdm(unique_labels, desc="Seg resampling", disable=False)):
        mask = seg == label
        reshaped_multihot = functional.interpolate(mask.float(), target_shape, mode='trilinear')
        reshaped[reshaped_multihot >= 0.5] = label
    return reshaped

def resize_segmentation_tensor_parallel(seg: torch.Tensor, target_shape, processes) -> torch.Tensor:
    target_shape_tmp = np.asarray(seg.shape)
    target_shape_tmp[2:] = np.asarray(target_shape)
    target_shape_tmp = tuple(target_shape_tmp)
    reshaped = torch.zeros(target_shape_tmp, dtype=seg.dtype, device=seg.device)
    unique_labels = torch.unique(seg, sorted=True)

    reshaped = imap_tqdm2(resize_segmentation_tensor_parallel_single, unique_labels, processes, reshaped, desc="Seg resampling", seg=seg, target_shape=target_shape)

    return reshaped


def resize_segmentation_tensor_parallel_single(label, seg, target_shape):
    mask = seg == label
    reshaped_multihot = functional.interpolate(mask.float(), target_shape, mode='trilinear')
    return reshaped_multihot


def imap_tqdm2(function, iterable, processes, reshaped, chunksize=1, desc=None, disable=False, **kwargs):
    """
    Run a function in parallel with a tqdm progress bar and an arbitrary number of arguments.
    Results are always ordered and the performance should be the same as of Pool.map.
    TODO: Still needs more performance testing
    :param function: The function that should be parallelized.
    :param iterable: The iterable passed to the function.
    :param processes: The number of processes used for the parallelization.
    :param chunksize: The iterable is based on the chunk size chopped into chunks and submitted to the process pool as separate tasks.
    :param desc: The description displayed by tqdm in the progress bar.
    :param disable: Disables the tqdm progress bar.
    :param kwargs: Any additional arguments that should be passed to the function.
    """
    if kwargs:
        function_wrapper = partial(wrapper, function=function, **kwargs)
    else:
        function_wrapper = partial(wrapper, function=function)

    with Pool(processes=processes) as p:
        with tqdm(desc=desc, total=len(iterable), disable=disable) as pbar:
            for i, reshaped_multihot, label in p.imap_unordered(function_wrapper, enumerate(iterable), chunksize=chunksize):
                reshaped[reshaped_multihot >= 0.5] = label
                pbar.update()
    return reshaped


def wrapper(enum_iterable, function, **kwargs):
    i = enum_iterable[0]
    result = function(enum_iterable[1], **kwargs)
    return i, result, enum_iterable[1]


def standardize(img_npy: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        img_npy = torch.from_numpy(img_npy.astype(np.float32)).cuda()
        mn = img_npy.mean()
        sd = img_npy.std()
        img_npy -= mn
        img_npy /= sd

    img_npy = img_npy.cpu().numpy()
    torch.cuda.empty_cache()
    return img_npy


def standardize_tio(subject, zscore, gpu=True):
    image = subject["image"].tensor
    with torch.no_grad():
        image = image.float()
        if gpu:
            image = image.cuda()
        # mn = image.mean()
        # sd = image.std()
        image -= zscore["mean"]
        image /= zscore["std"]

    subject["image"] = tio.ScalarImage(tensor=image.cpu())
    torch.cuda.empty_cache()
    return subject


import numpy as np
from os.path import join
from natsort import natsorted
import os

def load_filepaths(load_dir, extension=None, return_path=True, return_extension=True):
    filepaths = []
    if isinstance(extension, str):
        extension = tuple([extension])
    elif isinstance(extension, list):
        extension = tuple(extension)
    elif extension is not None and not isinstance(extension, tuple):
        raise RuntimeError("Unknown type for argument extension.")

    if extension is not None:
        extension = list(extension)
        for i in range(len(extension)):
            if extension[i][0] != ".":
                extension[i] = "." + extension[i]
        extension = tuple(extension)

    for filename in os.listdir(load_dir):
        if extension is None or str(filename).endswith(extension):
            if not return_extension:
                if extension is None:
                    filename = filename.split(".")[0]
                else:
                    for ext in extension:
                        if str(filename).endswith((ext)):
                            filename = str(filename)[:-len(ext)]
            if return_path:
                filename = join(load_dir, filename)
            filepaths.append(filename)
    filepaths = np.asarray(filepaths)
    filepaths = natsorted(filepaths)

    return filepaths


def load_nifti(filename, return_meta=False, is_seg=False):
    image = sitk.ReadImage(filename)
    image_np = sitk.GetArrayFromImage(image)

    if is_seg:
        image_np = np.rint(image_np)
        # image_np = image_np.astype(np.int16)  # In special cases segmentations can contain negative labels, so no np.uint8

    if not return_meta:
        return image_np
    else:
        spacing = image.GetSpacing()
        keys = image.GetMetaDataKeys()
        header = {key:image.GetMetaData(key) for key in keys}
        affine = None  # How do I get the affine transform with SimpleITK? With NiBabel it is just image.affine
        return image_np, spacing, affine, header


def save_nifti(filename, image, spacing=None, affine=None, header=None, is_seg=False, dtype=None, in_background=False):
    if is_seg:
        image = np.rint(image)
        if dtype is None:
            image = image.astype(np.int16)  # In special cases segmentations can contain negative labels, so no np.uint8 by default

    if dtype is not None:
        image = image.astype(dtype)

    image = sitk.GetImageFromArray(image)

    if header is not None:
        [image.SetMetaData(key, header[key]) for key in header.keys()]

    if spacing is not None:
        image.SetSpacing(spacing)

    if affine is not None:
        pass  # How do I set the affine transform with SimpleITK? With NiBabel it is just nib.Nifti1Image(img, affine=affine, header=header)

    if not in_background:
        sitk.WriteImage(image, filename)
    else:
        global_pool, global_pool_results = global_mp_pool.get_pool()
        # global_pool_results.append(global_pool.starmap_async(_save, ((filename, image), )))
        global_mp_pool.queue_job(global_pool.starmap_async, _save, ((filename, image), ))


def _save(filename, image):
    sitk.WriteImage(image, filename)


def normalize(x, x_min=None, x_max=None):
    if x_min is None:
        x_min = x.min()

    if x_max is None:
        x_max = x.max()

    if x_min == x_max:
        return x * 0
    else:
        return (x - x.min()) / (x.max() - x.min())
