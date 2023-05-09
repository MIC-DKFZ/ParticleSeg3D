import sys
from pathlib import Path
sys.path.append(str(Path('').absolute().parent))

import torchio as tio
import numpy as np
from tqdm import tqdm
from particle_seg.helper import utils
from particle_seg.helper.imap_tqdm import imap_tqdm
import json
from os.path import join
import gc
import SimpleITK as sitk
from nnunet.dataset_conversion.utils import generate_dataset_json
from sampler import GridSampler, MultiSizeUniformSampler
from particle_seg.conversion.instance2border_semantic import instance2border_semantic_process
import zarr
from particle_touch_preprocess import preprocess_seg_all


def preprocess_all(load_dir, metadata_load_filepath, zscore_load_filepath, names, save_dir,
                   target_spacing, target_particle_size_in_pixel, target_patch_size_in_pixel, standardize, dataset_name, train, processes, auto_scale, multi_size, compress,
                   border_thickness_in_pixel, zscore, resample_processes):
    with open(metadata_load_filepath) as f:
        metadata = json.load(f)

    with open(zscore_load_filepath) as f:
        zscore_metadata = json.load(f)

    if train:
        image_save_dir = join(save_dir, dataset_name, "imagesTr")
        semantic_seg_save_dir = join(save_dir, dataset_name, "labelsTr")
        instance_seg_save_dir = join(save_dir, dataset_name, "labelsTr_instance")
        semantic_seg_zarr_save_dir = join(save_dir, dataset_name, "labelsTr_zarr")
        instance_seg_zarr_save_dir = join(save_dir, dataset_name, "labelsTr_instance_zarr")
        Path(instance_seg_save_dir).mkdir(parents=True, exist_ok=True)
    else:
        image_save_dir = join(save_dir, dataset_name, "images")
        semantic_seg_save_dir = join(save_dir, dataset_name, "predictions")
        instance_seg_save_dir = None
        semantic_seg_zarr_save_dir = None
        instance_seg_zarr_save_dir = None
    Path(image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(semantic_seg_save_dir).mkdir(parents=True, exist_ok=True)

    for name in names:
        if name not in metadata:
            raise RuntimeError("{} is missing in metadata!".format(name))

    # patch_metadata = {}

    # for name in names:
    #     image_load_filepath = join(load_dir, "images", name + ".nii.gz")
    #     seg_load_filepath = None
    #     if train:
    #         seg_load_filepath = join(load_dir, "instance_seg", name + ".nii.gz")
    #
    #     patch_metadata_single = preprocess_single(image_load_filepath, seg_load_filepath, metadata_load_filepath, zscore_metadata,
    #                                               image_save_dir, semantic_seg_save_dir, instance_seg_save_dir, semantic_seg_zarr_save_dir, instance_seg_zarr_save_dir, name, train,
    #                                               target_spacing, target_particle_size_in_pixel, target_patch_size_in_pixel, standardize, parallel, auto_scale, multi_size, compress)
    #     patch_metadata[name] = patch_metadata_single

    image_load_filepaths = [join(load_dir, "images", name + ".nii.gz") for name in names]
    seg_load_filepaths = [join(load_dir, "instance_seg", name + ".nii.gz") for name in names]

    if processes is None:
        for i in tqdm(range(len(names))):
            _preprocess_single(i, names=names, image_load_filepaths=image_load_filepaths, seg_load_filepaths=seg_load_filepaths, metadata_load_filepath=metadata_load_filepath,
                      zscore_metadata=zscore_metadata, image_save_dir=image_save_dir, semantic_seg_save_dir=semantic_seg_save_dir, instance_seg_save_dir=instance_seg_save_dir,
                      semantic_seg_zarr_save_dir=semantic_seg_zarr_save_dir, instance_seg_zarr_save_dir=instance_seg_zarr_save_dir, train=train, target_spacing=target_spacing,
                      target_particle_size_in_pixel=target_particle_size_in_pixel, target_patch_size_in_pixel=target_patch_size_in_pixel, standardize=standardize,
                      auto_scale=auto_scale, multi_size=multi_size, compress=compress, border_thickness_in_pixel=border_thickness_in_pixel, zscore=zscore, resample_processes=resample_processes)
    else:
        imap_tqdm(_preprocess_single, range(len(names)), processes, names=names, image_load_filepaths=image_load_filepaths, seg_load_filepaths=seg_load_filepaths, metadata_load_filepath=metadata_load_filepath,
                  zscore_metadata=zscore_metadata, image_save_dir=image_save_dir, semantic_seg_save_dir=semantic_seg_save_dir, instance_seg_save_dir=instance_seg_save_dir,
                  semantic_seg_zarr_save_dir=semantic_seg_zarr_save_dir, instance_seg_zarr_save_dir=instance_seg_zarr_save_dir, train=train, target_spacing=target_spacing,
                  target_particle_size_in_pixel=target_particle_size_in_pixel, target_patch_size_in_pixel=target_patch_size_in_pixel, standardize=standardize,
                  auto_scale=auto_scale, multi_size=multi_size, compress=compress, border_thickness_in_pixel=border_thickness_in_pixel, zscore=zscore, resample_processes=resample_processes)

    if train:
        modality = "CT"
        if standardize:
            modality = "noNorm"
        generate_dataset_json(join(save_dir, dataset_name, 'dataset.json'), join(save_dir, dataset_name, "imagesTr"), None, (modality,), {0: 'bg', 1: 'core', 2: 'border'}, dataset_name)

        preprocess_seg_all(join(save_dir, dataset_name, "labelsTr_instance"), join(save_dir, dataset_name, "regionprops.pkl"))
    # else:
    #
    #     with open(join(save_dir, dataset_name, 'patch_metadata.json'), 'w', encoding='utf-8') as f:
    #         json.dump(patch_metadata, f, ensure_ascii=False, indent=4)


def _preprocess_single(i, names, image_load_filepaths, seg_load_filepaths, metadata_load_filepath, zscore_metadata,
                      image_save_dir, semantic_seg_save_dir, instance_seg_save_dir, semantic_seg_zarr_save_dir, instance_seg_zarr_save_dir, train,
                      target_spacing, target_particle_size_in_pixel, target_patch_size_in_pixel, standardize, auto_scale, multi_size, compress, border_thickness_in_pixel, zscore, resample_processes):
    name = names[i]
    image_load_filepath = image_load_filepaths[i]
    seg_load_filepath = seg_load_filepaths[i]

    preprocess_single(image_load_filepath, seg_load_filepath, metadata_load_filepath, zscore_metadata,
                      image_save_dir, semantic_seg_save_dir, instance_seg_save_dir, semantic_seg_zarr_save_dir, instance_seg_zarr_save_dir, name, train,
                      target_spacing, target_particle_size_in_pixel, target_patch_size_in_pixel, standardize, auto_scale, multi_size, compress, border_thickness_in_pixel, zscore=zscore, resample_processes=resample_processes)


def preprocess_single(image_load_filepath, seg_load_filepath, metadata_load_filepath, zscore_metadata,
                      image_save_dir, semantic_seg_save_dir, instance_seg_save_dir, semantic_seg_zarr_save_dir, instance_seg_zarr_save_dir, name, train,
                      target_spacing, target_particle_size_in_pixel, target_patch_size_in_pixel, standardize, auto_scale, multi_size, compress, border_thickness_in_pixel=3,
                      disable_sampling=True, zscore="global_zscore", resample_processes=None, disable_particle_normalization=False):  # border_thickness_in_pixel=5
    with open(metadata_load_filepath) as f:
        metadata = json.load(f)

    print("Processing: ", name)
    if train:
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=sitk.GetArrayFromImage(sitk.ReadImage(image_load_filepath))[np.newaxis, ...]),
            label=tio.LabelMap(tensor=sitk.GetArrayFromImage(sitk.ReadImage(seg_load_filepath))[np.newaxis, ...])
        )
    else:
        subject = tio.Subject(
            image=tio.ScalarImage(tensor=sitk.GetArrayFromImage(sitk.ReadImage(image_load_filepath))[np.newaxis, ...])
        )

    if standardize:
        zscore = zscore_metadata["global_zscore"] if zscore == "global_zscore" else zscore_metadata["local_zscore"][name]
        subject = utils.standardize_tio(subject, zscore, gpu=not disable_sampling)

    image_shape = subject["image"].shape[1:]
    source_particle_size_in_mm = tuple(metadata[name]["particle_size"])
    source_spacing = tuple(metadata[name]["spacing"])
    target_particle_size_in_mm = tuple(pixel2mm(target_particle_size_in_pixel, target_spacing))

    patch_metadata = {}
    if not disable_sampling:
        target_patch_size_in_pixel, source_patch_size_in_pixel, size_conversion_factor = compute_patch_size(subject, name, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel,
                                                                                    source_spacing, source_particle_size_in_mm, image_shape, auto_scale)
        sampler = init_sampler(subject, name, target_spacing, target_patch_size_in_pixel,
                               source_spacing, source_particle_size_in_mm, source_patch_size_in_pixel, image_shape, auto_scale, multi_size)
        patch_metadata["original_image_size"] = subject["image"].shape[1:]
        patch_metadata["source_spacing"] = tuple([value for value in source_spacing])
        patch_metadata["source_particle_size_in_mm"] = tuple([value for value in source_particle_size_in_mm])
        patch_metadata["source_particle_size_in_pixel"] = tuple([value for value in mm2pixel(source_particle_size_in_mm, source_spacing)])
        patch_metadata["source_patch_size_in_pixel"] = tuple([int(value) for value in source_patch_size_in_pixel])
        patch_metadata["target_spacing"] = tuple([value for value in target_spacing])
        patch_metadata["target_particle_size_in_mm"] = tuple([value for value in target_particle_size_in_mm])
        patch_metadata["target_patch_size_in_pixel"] = tuple([value for value in target_patch_size_in_pixel])
        patch_metadata["target_particle_size_in_pixel"] = tuple([value for value in target_particle_size_in_pixel])
    else:
        size_conversion_factor = compute_size_conversion_factor(source_particle_size_in_mm, source_spacing, target_particle_size_in_mm, target_spacing)
        target_patch_size_in_pixel = np.rint(np.asarray(image_shape) / size_conversion_factor).astype(int)
        target_patch_size_in_pixel = target_patch_size_in_pixel.tolist()
        print("target_patch_size_in_pixel: ", target_patch_size_in_pixel)
        print("size_conversion_factor: ", size_conversion_factor)

        subject["location"] = ((0, 0), (0, 0), (0, 0))
        sampler = [subject]

    print("original_image_size: ", subject["image"].shape[1:])
    print("source_spacing: ", source_spacing)
    print("source_particle_size_in_mm: ", tuple([value for value in source_particle_size_in_mm]))
    print("source_particle_size_in_pixel: ", tuple([value for value in mm2pixel(source_particle_size_in_mm, source_spacing)]))
    print("source_patch_size_in_pixel: ", "---")
    print("size_conversion_factor: ", size_conversion_factor)
    print("target_spacing: ", tuple([value for value in target_spacing]))
    print("target_particle_size_in_mm: ", tuple([value for value in target_particle_size_in_mm]))
    print("target_patch_size_in_pixel: ", tuple([value for value in target_patch_size_in_pixel]))
    print("target_particle_size_in_pixel: ", tuple([value for value in target_particle_size_in_pixel]))

    # If train and multi_size is true, then these values will be wrong. That is why we don't write them to disk.
    patch_metadata["patches"] = {}

    for i, patch in enumerate(tqdm(sampler, total=len(sampler), desc=name, disable=True)):
        print("Resampling...")
        if not disable_particle_normalization:
            patch = utils.resample_tio(patch, target_patch_size_in_pixel, gpu=not disable_sampling, processes=resample_processes)
        image = patch["image"].numpy()[0]
        if np.sum(np.nonzero(image)) <= 5:
            continue
        patch_name = "{}_{}".format(name, str(i).zfill(5))
        image_save_filepath = join(image_save_dir, patch_name + "_0000.nii.gz")
        if not compress:
            utils.save_nifti(image_save_filepath, image, spacing=target_spacing)
        else:
            contrast_limits = metadata[name]["contrast"]
            image = np.clip(image, contrast_limits[0], contrast_limits[1])
            image = (utils.normalize(image) * 255).astype(np.uint8)
            utils.save_nifti(image_save_filepath, image, spacing=target_spacing, dtype=np.uint8)
        if train:
            instance_seg = patch["label"].numpy()[0]
            semantic_seg = instance2border_semantic_process(instance_seg, border_thickness_in_pixel=border_thickness_in_pixel)
            semantic_seg_save_filepath = join(semantic_seg_save_dir, patch_name + ".nii.gz")
            instance_seg_save_filepath = join(instance_seg_save_dir, patch_name + ".nii.gz")
            semantic_seg_zarr_save_filepath = join(semantic_seg_zarr_save_dir, patch_name + ".zarr")
            instance_seg_zarr_save_filepath = join(instance_seg_zarr_save_dir, patch_name + ".zarr")
            utils.save_nifti(semantic_seg_save_filepath, semantic_seg, spacing=target_spacing, is_seg=True, dtype=np.uint8)
            utils.save_nifti(instance_seg_save_filepath, instance_seg, spacing=target_spacing, is_seg=True, dtype=np.uint16)
            semantic_seg = zarr.array(semantic_seg)
            zarr.save(semantic_seg_zarr_save_filepath, semantic_seg, chunks=(64, 64, 64))
            instance_seg = zarr.array(instance_seg)
            zarr.save(instance_seg_zarr_save_filepath, instance_seg, chunks=(64, 64, 64))
        location = patch["location"]
        if not isinstance(location, tuple):
            location = location.numpy().tolist()
        patch_metadata["patches"][i] = {"patch_name": patch_name, "patch_location": location}

    del sampler
    del subject
    gc.collect()

    return patch_metadata


def compute_patch_size(subject, name, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel, source_spacing, source_particle_size_in_mm, image_shape, auto_scale):
    size_conversion_factor = compute_size_conversion_factor(source_particle_size_in_mm, source_spacing, target_particle_size_in_mm, target_spacing)
    source_patch_size_in_pixel = np.rint(target_patch_size_in_pixel * size_conversion_factor).astype(int)

    too_large = False
    if image_shape[0] < source_patch_size_in_pixel[0] or image_shape[1] < source_patch_size_in_pixel[1] or image_shape[2] < source_patch_size_in_pixel[2]:
        max_index = np.argmax(np.asarray(source_patch_size_in_pixel) - np.asarray(image_shape))
        max_target_patch_size_in_pixel = (image_shape[max_index] / source_patch_size_in_pixel[max_index]) * np.asarray(target_patch_size_in_pixel)
        max_target_patch_size_in_pixel = np.floor(max_target_patch_size_in_pixel).astype(int)
        target_patch_size_in_pixel_old = target_patch_size_in_pixel
        source_patch_size_in_pixel_old = source_patch_size_in_pixel
        target_patch_size_in_pixel = tuple([int(value) for value in max_target_patch_size_in_pixel])
        source_patch_size_in_pixel = np.rint(target_patch_size_in_pixel * size_conversion_factor).astype(int)
        too_large = True

    if not auto_scale and too_large:
        print(
            "WARNING: The source_patch_size_in_pixel of {} is too large. Maximum possible target_patch_size_in_pixel is {}, but current target_patch_size_in_pixel is {}. target_patch_size_in_pixel will be set to {} for {}.".format(
                source_patch_size_in_pixel_old, max_target_patch_size_in_pixel, target_patch_size_in_pixel_old, max_target_patch_size_in_pixel, name))

    # if auto_scale:
    #     source_patch_size_in_pixel = auto_scale_num_patches(subject, source_spacing, source_particle_size_in_mm, source_patch_size_in_pixel, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel)

    print("Image shape: {}, source_spacing: {}, source_particle_size_in_mm: {}, source_particle_size_in_pixel: {}, source_patch_size_in_mm: {}, source_patch_size_in_pixel: {}, size_conversion_factor: {}".format(
        image_shape, source_spacing, source_particle_size_in_mm, mm2pixel(source_particle_size_in_mm, source_spacing), pixel2mm(source_patch_size_in_pixel, source_spacing), source_patch_size_in_pixel,
        size_conversion_factor
    ))

    return target_patch_size_in_pixel, source_patch_size_in_pixel, size_conversion_factor


# def auto_scale_num_patches(subject, source_spacing, source_particle_size_in_mm, source_patch_size_in_pixel,
#                             target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel, optimal_num_patches=30, min_particle_size_in_pixel=30, max_particle_size_in_pixel=100):
#     print("Auto scaling number of patches to extract...")
#     sampler, n_samples = create_sampler(subject, source_patch_size_in_pixel)
#     target_particle_size_in_pixel = mm2pixel(target_particle_size_in_mm, target_spacing)
#     too_many_patches = True
#     if n_samples < optimal_num_patches:
#         too_many_patches = False
#     while n_samples != optimal_num_patches:
#         if too_many_patches and n_samples > optimal_num_patches:
#             if target_particle_size_in_pixel - 5 < min_particle_size_in_pixel:
#                 break
#             target_particle_size_in_pixel -= 5
#         else:
#             if max_particle_size_in_pixel + 5 > max_particle_size_in_pixel:
#                 break
#             target_particle_size_in_pixel += 5
#
#         target_particle_size_in_mm = tuple(pixel2mm(target_particle_size_in_pixel, target_spacing))
#         size_conversion_factor = compute_size_conversion_factor(source_particle_size_in_mm, source_spacing, target_particle_size_in_mm, target_spacing)
#         source_patch_size_in_pixel = np.rint(target_patch_size_in_pixel * size_conversion_factor).astype(int)
#
#         sampler, n_samples = create_sampler(subject, source_patch_size_in_pixel)
#
#         if (too_many_patches and n_samples <= optimal_num_patches) or( not too_many_patches and n_samples >= optimal_num_patches):
#             break
#
#     print("Finished auto scaling.")
#     return source_patch_size_in_pixel


def init_sampler(subject, name, target_spacing, target_patch_size_in_pixel,
                           source_spacing, source_particle_size_in_mm, source_patch_size_in_pixel, image_shape, auto_scale, multi_size):
    if not multi_size:
        return GridSampler(subject, source_patch_size_in_pixel)
    else:
        min_target_particle_size_in_pixel = 60  # 30
        max_target_particle_size_in_pixel = 60  # 100
        num_patches = 10  # 50
        target_particle_size_in_mm = tuple(pixel2mm(min_target_particle_size_in_pixel, target_spacing))
        _, max_patch_size, _ = compute_patch_size(subject, name, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel,
                                                                                    source_spacing, source_particle_size_in_mm, image_shape, auto_scale)
        target_particle_size_in_mm = tuple(pixel2mm(max_target_particle_size_in_pixel, target_spacing))
        _, min_patch_size, _ = compute_patch_size(subject, name, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel,
                                                                                    source_spacing, source_particle_size_in_mm, image_shape, auto_scale)
        for i in range(len(max_patch_size.shape)):
            if max_patch_size[i] < min_patch_size[i]:  # Sometimes max_patch_size == min_patch_size, but min_patch_size can be slightly greater than max_patch_size due to rounding errors
                max_patch_size[i] = min_patch_size[i]
        return MultiSizeUniformSampler(subject, min_patch_size, max_patch_size, num_patches)


def pixel2mm(length, spacing):
    return np.asarray(length) * np.asarray(spacing)


def mm2pixel(length, spacing):
    return np.asarray(length) / np.asarray(spacing)


def compute_size_conversion_factor(source_particle_size_in_mm, source_spacing, target_particle_size_in_mm, target_spacing):
    factor = np.asarray(target_spacing) / np.asarray(source_spacing)
    factor *= np.asarray(source_particle_size_in_mm) / np.asarray(target_particle_size_in_mm)
    return factor
