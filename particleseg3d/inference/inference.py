from torch.utils.data import DataLoader
from particleseg3d.utils import utils
import pytorch_lightning as pl
from os.path import join
from numcodecs import blosc
import shutil
import zarr
from particleseg3d.inference.sampler import SamplerDataset, GridSampler, ResizeSampler, ChunkedGridSampler, ChunkedResizeSampler
from particleseg3d.inference.aggregator import WeightedSoftmaxAggregator, ResizeChunkedWeightedSoftmaxAggregator
import numpy as np
from tqdm import tqdm
from model_nnunet import Nnunet
import torch
from particleseg3d.inference.size_conversion import compute_patch_size, pixel2mm, mm2pixel
import json
from particleseg3d.inference.border_core2instance import border_core2instance
from skimage import transform as ski_transform
from pathlib import Path
import argparse
import pickle
import time
from batchgenerators.augmentations.utils import pad_nd_image
import cc3d
import numpy_indexed as npi


def setup_model(experiment_dir, folds, nnunet_trainer, configuration, reuse):
    with open(join(experiment_dir, "plans.pkl"), 'rb') as handle:
        config = pickle.load(handle)

    model = None
    trainer = None
    if not reuse:
        model = Nnunet(experiment_dir, folds=folds, nnunet_trainer=nnunet_trainer, configuration=configuration)
        # model.network.load_state_dict(torch.load(checkpoint_path)["state_dict"])
        model.eval()
        trainer = pl.Trainer(gpus=1, precision=16)
    return trainer, model, config


def predict_cases(load_dir, save_dir, names, metadata_load_filepath, trainer, model, config, target_particle_size, target_spacing, processes, trainer_name, min_rel_particle_size, reuse):
    if names is None:
        names = utils.load_filepaths(load_dir, return_path=False, return_extension=False)

    for name in tqdm(names, desc="Inference Query"):
        predict_case(load_dir, save_dir, name, metadata_load_filepath, trainer, model, config, target_particle_size, target_spacing, processes, trainer_name, min_rel_particle_size, reuse)


def predict_case(load_dir, save_dir, name, metadata_load_filepath, trainer, model, config, target_particle_size_in_pixel, target_spacing, processes, trainer_name, min_rel_particle_size, reuse):
    print("Starting inference of sample: ", name)
    load_filepath = join(load_dir, "{}.zarr".format(name))
    pred_softmax_filepath, pred_border_core_filepath, pred_border_core_tmp_filepath, pred_instance_filepath = setup_folder_structure(save_dir, name, reuse)

    with open(metadata_load_filepath) as f:
        metadata = json.load(f)

    target_particle_size_in_mm = pixel2mm(target_particle_size_in_pixel, target_spacing)
    target_patch_size_in_pixel = np.asarray(list(config['plans_per_stage'].values())[-1]['patch_size'])
    source_particle_size = metadata[name]["particle_size"]
    source_spacing = metadata[name]["spacing"]

    predict(load_filepath, pred_softmax_filepath, pred_border_core_filepath, pred_border_core_tmp_filepath, pred_instance_filepath, target_spacing, target_particle_size_in_mm, target_particle_size_in_pixel, target_patch_size_in_pixel,
            source_spacing, source_particle_size, trainer, model, reuse, processes, trainer_name, min_rel_particle_size)


def setup_folder_structure(save_dir, name, reuse):
    Path(join(save_dir, name)).mkdir(parents=True, exist_ok=True)
    pred_softmax_filepath = join(save_dir, name, "{}_softmax_tmp.zarr".format(name))
    pred_border_core_filepath = join(save_dir, name, "{}_border.zarr".format(name))
    pred_border_core_tmp_filepath = join(save_dir, name, "{}_border_tmp.zarr".format(name))
    pred_instance_filepath = join(save_dir, name, "{}".format(name))
    if not reuse:
        shutil.rmtree(pred_softmax_filepath, ignore_errors=True)
        shutil.rmtree(pred_border_core_filepath, ignore_errors=True)
        shutil.rmtree(pred_border_core_tmp_filepath, ignore_errors=True)
        shutil.rmtree(pred_instance_filepath, ignore_errors=True)
    return pred_softmax_filepath, pred_border_core_filepath, pred_border_core_tmp_filepath, pred_instance_filepath


def predict(load_filepath, pred_softmax_filepath, pred_border_core_filepath, pred_border_core_tmp_filepath, pred_instance_filepath, target_spacing, target_particle_size_in_mm, target_particle_size_in_pixel, target_patch_size_in_pixel,
            source_spacing, source_particle_size, trainer, model, reuse, processes, trainer_name, min_rel_particle_size, postprocessing=False):
    try:
        img = zarr.open(load_filepath, mode='r')
    except zarr.errors.PathNotFoundError as e:
        print("Filepath: ", load_filepath)
        raise e
    # print(sorted(img.attrs))
    # crop_bbox = img.attrs["bbox"]
    # original_size_before_crop = img.attrs["original_size"]
    if postprocessing:
        with open(load_filepath[:-5] + ".pkl", 'rb') as handle:
            properties = pickle.load(handle)
        crop_bbox = properties["bbox"]
        original_size_before_crop = properties["original_size"]
    else:
        crop_bbox = None
        original_size_before_crop = None

    if not reuse:
        source_patch_size_in_pixel, source_chunk_size, resized_image_shape, resized_chunk_size = compute_zoom(img, source_spacing, source_particle_size, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel)

        # original_image_shape = img.shape
        img, crop_slices = pad_image(img, source_patch_size_in_pixel)
        # resized_image_shape_backup = resized_image_shape
        # if image_shape != img.shape:
        #     resized_image_shape = img.shape

        source_patch_size_in_pixel, source_chunk_size, resized_image_shape, resized_chunk_size = compute_zoom(img, source_spacing, source_particle_size, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel)

        sampler, aggregator, chunked = create_sampler_and_aggregator(img, pred_border_core_filepath, source_patch_size_in_pixel, target_patch_size_in_pixel, resized_image_shape, source_chunk_size, resized_chunk_size, target_spacing, trainer_name)

        model.prediction_setup(aggregator, chunked)
        print("Inference...")
        trainer.predict(model, dataloaders=sampler)
        border_core_resized_pred = aggregator.get_output()
        shutil.rmtree(pred_softmax_filepath, ignore_errors=True)
    else:
        # border_core_resized_pred = zarr.open(pred_border_core_filepath, mode='r')
        source_patch_size_in_pixel, source_chunk_size, resized_image_shape, resized_chunk_size = compute_zoom(img, source_spacing, source_particle_size, target_spacing, target_particle_size_in_mm,
                                                                                                              target_patch_size_in_pixel)
        img, crop_slices = pad_image(img, source_patch_size_in_pixel)
        border_core_resized_pred = utils.load_nifti(pred_instance_filepath + "_border_core_zoomed.nii.gz")

    instance_pred = border_core2instance_conversion(border_core_resized_pred, pred_border_core_tmp_filepath, crop_slices, img.shape, crop_bbox, original_size_before_crop, target_spacing, source_spacing, target_particle_size_in_pixel, pred_instance_filepath, postprocessing=postprocessing, processes=processes, reuse=reuse)
    instance_pred = filter_small_particles(instance_pred, min_rel_particle_size)
    save_prediction(instance_pred, pred_instance_filepath, source_spacing)

    shutil.rmtree(pred_border_core_filepath, ignore_errors=True)
    if not reuse:
        shutil.rmtree(pred_border_core_tmp_filepath, ignore_errors=True)


def compute_zoom(img, source_spacing, source_particle_size, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel):
    if np.array_equal(target_particle_size_in_mm, [0, 0, 0]):
        return target_patch_size_in_pixel, target_patch_size_in_pixel * 4, img.shape, target_patch_size_in_pixel * 4
    image_shape = np.asarray(img.shape[-3:])
    source_particle_size_in_mm = tuple(source_particle_size)
    source_spacing = tuple(source_spacing)
    _, source_patch_size_in_pixel, size_conversion_factor = compute_patch_size(target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel, source_spacing, source_particle_size_in_mm, image_shape)
    for i in range(len(source_patch_size_in_pixel)):
        if source_patch_size_in_pixel[i] % 2 != 0:  # If source_patch_size_in_pixel is odd then patch_overlap is not a round number. This fixes that.
            source_patch_size_in_pixel[i] += 1
    size_conversion_factor = (target_patch_size_in_pixel / source_patch_size_in_pixel)
    resized_image_shape = np.rint(image_shape * size_conversion_factor).astype(np.int32)
    if np.any(source_patch_size_in_pixel * 4 > image_shape):
        source_chunk_size = source_patch_size_in_pixel * 2
    else:
        source_chunk_size = source_patch_size_in_pixel * 4
    resized_chunk_size = np.rint(source_chunk_size * size_conversion_factor).astype(np.int32)
    print("source_spacing: ", source_spacing)
    print("target_spacing: ", target_spacing)
    print("source_particle_size_in_mm: ", source_particle_size_in_mm)
    print("source_particle_size_in_pixel: ", mm2pixel(source_particle_size_in_mm, source_spacing))
    print("target_particle_size_in_mm: ", target_particle_size_in_mm)
    print("target_particle_size_in_pixel: ", mm2pixel(target_particle_size_in_mm, target_spacing))
    print("size_conversion_factor: ", size_conversion_factor)
    print("source_patch_size_in_pixel: ", source_patch_size_in_pixel)
    print("target_patch_size_in_pixel: ", target_patch_size_in_pixel)
    print("image_shape: ", img.shape)
    print("resized_image_shape: ", resized_image_shape)
    print("source_chunk_size: ", source_chunk_size)
    print("resized_chunk_size: ", resized_chunk_size)
    return source_patch_size_in_pixel, source_chunk_size, resized_image_shape, resized_chunk_size


def create_sampler_and_aggregator(img, pred_border_core_filepath, source_patch_size_in_pixel, target_patch_size_in_pixel, resized_image_shape, source_chunk_size, resized_chunk_size, target_spacing, trainer_name):
    region_class_order = None
    batch_size = 6
    num_workers = 12
    num_channels = 3
    if trainer_name == "nnUNetTrainerV2GlasRegions__nnUNetPlansv2.1":
        region_class_order = (1, 2)
        num_channels = 2
    if np.prod(resized_image_shape) < 1000*1000*500:
        print("Chunked: False")
        pred = zarr.open(pred_border_core_filepath, mode='w', shape=(num_channels, *resized_image_shape), chunks=(3, 64, 64, 64), dtype=np.float32)
        blosc.set_nthreads(4)
        sampler = GridSampler(img, image_size=img.shape[-3:], patch_size=source_patch_size_in_pixel, patch_overlap=source_patch_size_in_pixel // 2)
        print("Sampler length: ", len(sampler))
        if not np.array_equal(img.shape, resized_image_shape):
            sampler = ResizeSampler(sampler, target_size=target_patch_size_in_pixel, image_size=resized_image_shape[-3:], patch_size=target_patch_size_in_pixel, patch_overlap=target_patch_size_in_pixel // 2)
        sampler = SamplerDataset(sampler)
        sampler = DataLoader(sampler, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)
        aggregator = WeightedSoftmaxAggregator(pred, image_size=resized_image_shape[-3:], patch_size=target_patch_size_in_pixel, region_class_order=region_class_order)
        chunked = False
    else:
        print("Chunked: True")
        pred = zarr.open(pred_border_core_filepath, mode='w', shape=resized_image_shape[-3:], chunks=(64, 64, 64), dtype=np.uint8)
        blosc.set_nthreads(4)
        sampler = ChunkedGridSampler(img, image_size=img.shape[-3:], patch_size=source_patch_size_in_pixel, patch_overlap=source_patch_size_in_pixel // 2, chunk_size=source_chunk_size)
        print("Sampler length: ", len(sampler))
        if not np.array_equal(img.shape, resized_image_shape):
            sampler = ChunkedResizeSampler(sampler, target_size=target_patch_size_in_pixel, image_size=resized_image_shape[-3:], patch_size=target_patch_size_in_pixel, patch_overlap=target_patch_size_in_pixel // 2, chunk_size=resized_chunk_size)
        sampler = SamplerDataset(sampler)
        sampler = DataLoader(sampler, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)
        aggregator = ResizeChunkedWeightedSoftmaxAggregator(pred, image_size=resized_image_shape[-3:], patch_size=target_patch_size_in_pixel, patch_overlap=target_patch_size_in_pixel // 2, chunk_size=resized_chunk_size, spacing=target_spacing, region_class_order=region_class_order)
        chunked = True
    return sampler, aggregator, chunked


def border_core2instance_conversion(border_core_pred, pred_border_core_tmp_filepath, crop_slices, original_size, crop_bbox, original_size_before_crop,
                                    target_spacing, source_spacing, target_particle_size_in_pixel, save_filepath, debug=True, dtype=np.uint16, postprocessing=False, processes=None, reuse=False):
    print("Convert border-core prediction into instance prediction...")
    # prediction = np.array(prediction)
    if debug:
        border_core_pred_resampled = np.array(border_core_pred)
        # border_core_pred_resampled = ski_transform.resize(border_core_pred_resampled, output_shape=original_size, order=0, mode='edge', anti_aliasing=False)
        # if postprocessing:
        #     border_core_pred_resampled = postprocess(border_core_pred_resampled, crop_bbox, original_size_before_crop)
        utils.save_nifti(save_filepath + "_border_core_zoomed.nii.gz", border_core_pred_resampled, source_spacing)
    # prediction = border_semantic2instance_patchify(prediction, target_spacing)
    # prediction = border_semantic2instance_large_images(prediction, target_spacing)
    start_time = time.time()
    instance_pred, num_instances = border_core2instance(border_core_pred, pred_border_core_tmp_filepath, processes, dtype=dtype, reuse=False)
    print("Conversion time: {}s".format(round(time.time() - start_time), 2))
    start_time = time.time()
    print("Resampling prediction back to original image size...")
    # instance_pred = ski_transform.resize(instance_pred, output_shape=original_size, order=0, mode='edge')
    # instance_pred = ski_transform.resize(instance_pred, output_shape=original_size, order=1, mode='edge')
    # instance_pred = smooth_seg_resize(instance_pred, original_size, labels=list(range(num_instances+1)))
    # instance_pred = resample(instance_pred, original_size, is_seg=True)
    if debug:
        utils.save_nifti(save_filepath + "_zoomed.nii.gz", instance_pred, source_spacing)
    instance_pred = ski_transform.resize(instance_pred, original_size, 0, mode="edge", anti_aliasing=False)
    instance_pred = crop_pred(instance_pred, crop_slices)
    print("Resampling time: {}s".format(round(time.time() - start_time, 2)))
    if postprocessing:
        instance_pred = postprocess(instance_pred, crop_bbox, original_size_before_crop)
    return instance_pred


def filter_small_particles(instance_pred, min_rel_particle_size):
    print("Filter small particles...")
    if min_rel_particle_size is None:
        return instance_pred

    particle_voxels = cc3d.statistics(instance_pred)["voxel_counts"]
    particle_voxels = particle_voxels[1:]  # Remove background from list

    mean_particle_voxels = np.mean(particle_voxels) # median
    min_threshold = min_rel_particle_size * mean_particle_voxels
    print("min_rel_particle_size: ", min_rel_particle_size)
    print("min_threshold: ", min_threshold)

    instances_to_remove = np.arange(1, len(particle_voxels) + 1, dtype=int)
    instances_to_remove = instances_to_remove[particle_voxels < min_threshold]

    if len(instances_to_remove) > 0:
        target_values = np.zeros_like(instances_to_remove, dtype=int)
        shape = instance_pred.shape
        instance_pred = npi.remap(instance_pred.flatten(), instances_to_remove, target_values)
        instance_pred = instance_pred.reshape(shape)

    return instance_pred


def save_prediction(instance_pred, save_filepath, source_spacing, save_zarr=False):
    print("Saving prediction...")
    if save_zarr:
        instance_pred = zarr.creation.array(instance_pred, chunks=(64, 64, 64))
        instance_pred.attrs["spacing"] = source_spacing
        zarr.convenience.save(save_filepath + ".zarr", instance_pred)
    else:
        utils.save_nifti(save_filepath + ".nii.gz", instance_pred, source_spacing)


def resample(image: np.ndarray, target_shape, is_seg=False) -> np.ndarray:
    from torch.nn import functional
    if all([i == j for i, j in zip(image.shape, target_shape)]):
        return image

    with torch.no_grad():
        image = torch.from_numpy(image.astype(np.float32))
        if not is_seg:
            image = functional.interpolate(image[None, None], target_shape, mode='trilinear')[0, 0]
        else:
            image = functional.interpolate(image[None, None], target_shape, mode='nearest')[0, 0]
        image = image.numpy()
    torch.cuda.empty_cache()
    return image


def postprocess(prediction, crop_bbox, original_size):
    prediction_original_size = np.zeros(original_size)
    for c in range(3):
        crop_bbox[c][1] = np.min((crop_bbox[c][0] + prediction.shape[c], original_size[c]))
    prediction_original_size[crop_bbox[0][0]:crop_bbox[0][1], crop_bbox[1][0]:crop_bbox[1][1], crop_bbox[2][0]:crop_bbox[2][1]] = prediction
    return prediction_original_size


def pad_image(image, target_image_shape):
    # pad_width_after = np.asarray(target_image_shape) - np.asarray(image.shape)
    # pad_width_after = np.clip(pad_width_after, a_min=0, a_max=None)
    # pad_width_before = pad_width_after // 2
    # pad_width_after = pad_width_after - pad_width_before
    # pad_width_after = pad_width_after[..., np.newaxis]
    # pad_width_before = pad_width_before[..., np.newaxis]
    # pad_width = np.hstack((pad_width_before, pad_width_after))
    #
    # image = np.pad(image, pad_width, mode='constant', constant_values=0)
    # # image = np.pad(image, pad_width, mode='edge')

    if np.any(image.shape < target_image_shape):
        pad_kwargs = {'constant_values': 0}
        image = np.asarray(image)
        image, slices = pad_nd_image(image, target_image_shape, "constant", pad_kwargs, True, None)
        return image, slices
    else:
        return image, None


def crop_pred(pred, crop_slices):
    # target_pred_shape = np.asarray(target_pred_shape)[..., np.newaxis]
    # crop_indices = np.hstack((np.zeros_like(target_pred_shape), target_pred_shape))
    # pred = pred[slicer(pred, crop_indices)]
    if crop_slices is not None:
        pred = pred[tuple(crop_slices)]
    return pred


# def smooth_seg_resize(seg: np.ndarray, target_shape, order=1, labels=None, continuous=True, progressbar=True) -> np.ndarray:
#     """Order should be between 1-3. The higher the smoother, but also longer."""
#     reshaped = np.zeros(target_shape, dtype=seg.dtype)
#     if labels is None:
#         if continuous:
#             labels = list(range(np.max(seg) + 1))
#         else:
#             labels = np.unique(seg)
#
#     for i, label in enumerate(tqdm(labels, desc="Smooth Resampling", disable=not progressbar)):
#         mask = seg == label
#         reshaped_multihot = ski_transform.resize(mask.astype(float), target_shape, order, mode="edge", clip=True, anti_aliasing=False)
#         reshaped[reshaped_multihot >= 0.5] = label
#     return reshaped


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the base folder that contains the image as Zarr.")
    parser.add_argument('-m', "--metadata", required=True, help="Absolute path to the metadata.json.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the save folder.")
    parser.add_argument('-n', "--name", required=False, type=str, nargs="+", help="The name(s) without extension of the image(s) that should be used for inference. Multiple names must be separated by spaces.")
    parser.add_argument('-t', "--task", required=False, default=310, type=int, help="The task ID.")
    parser.add_argument('-tr', "--trainer", required=False, type=str, default="nnUNetTrainerV2_slimDA5_touchV5__nnUNetPlansv2.1", help="The trainer name.")
    parser.add_argument('-c', "--configuration", required=False, type=str, default="3D", help="The configuration 3D or 2D")
    parser.add_argument('-target_particle_size', default=(60, 60, 60), required=False, type=float, nargs=3,
                        help="The target particle size in pixels given as three numbers separate by spaces.")
    parser.add_argument('-target_spacing', default=(0.1, 0.1, 0.1), required=False, type=float, nargs=3,
                        help="The target spacing in millimeters given as three numbers separate by spaces.")
    parser.add_argument('-f', "--fold", required=False, default=(0, 1, 2, 3, 4), type=int, nargs="+", help="The folds to use. 0, 1, 2, 3, 4 or a combination.")
    parser.add_argument("--local", required=False, default=False, action='store_true', help="If inference is run on the workstation and not the cluster.")
    parser.add_argument('-p', '--processes', required=False, default=12, type=int, help="Number of processes to use for parallel processing. None to disable multiprocessing.")
    parser.add_argument("-min_rel_particle_size", required=False, default=0.005, type=float, help="Minimum relative particle size used for filtering.")
    parser.add_argument("--reuse", required=False, default=False, action='store_true', help="Reuse the border-core prediction.")
    args = parser.parse_args()

    print("Names: ", args.name)

    nnunet_configuration = "3d_fullres"
    if args.configuration == "2D":
        nnunet_configuration = "2d"

    # 206: local, random patches | 207: local, entire patches | 208: global, random patches | 209: global, entire patches
    experiment_dir = "/home/k539i/Documents/experiments/nnUNet/{}/Task{}_particle_seg/{}".format(nnunet_configuration, args.task, args.trainer)
    # if args.local:
    #     experiment_dir = "/home/k539i/Documents/experiments/nnUNet/{}/Task{}_particle_seg/{}".format(nnunet_configuration, args.task, args.trainer)
    # else:
    #     experiment_dir = "/dkfz/cluster/gpu/checkpoints/OE0441/k539i/nnUNet/{}/Task{}_particle_seg/{}".format(nnunet_configuration, args.task, args.trainer)
    print("experiment_dir: ", experiment_dir)

    trainer, model, config = setup_model(experiment_dir, args.fold, args.trainer, args.configuration, args.reuse)
    predict_cases(args.input, args.output, args.name, args.metadata, trainer, model, config, args.target_particle_size, args.target_spacing, args.processes, args.trainer, args.min_rel_particle_size, args.reuse)
