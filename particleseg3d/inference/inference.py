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
import json
from particleseg3d.inference.border_core2instance import border_core2instance
from skimage import transform as ski_transform
from pathlib import Path
import argparse
import pickle
from batchgenerators.augmentations.utils import pad_nd_image
import cc3d
import numpy_indexed as npi


def setup_model(model_dir, folds):
    with open(join(model_dir, "plans.pkl"), 'rb') as handle:
        config = pickle.load(handle)

    model = Nnunet(model_dir, folds=folds, nnunet_trainer="nnUNetTrainerV2_slimDA5_touchV5__nnUNetPlansv2.1", configuration="3d_fullres")
    model.eval()
    trainer = pl.Trainer(gpus=1, precision=16)
    return trainer, model, config


def predict_cases(load_dir, save_dir, names, trainer, model, config, target_particle_size, target_spacing, processes, min_rel_particle_size, zscore_norm):
    metadata_filepath = join(load_dir, "metadata.json")
    zscore_filepath = join(load_dir, "zscore.json")

    if names is None:
        names = utils.load_filepaths(load_dir, return_path=False, return_extension=False)

    for name in tqdm(names, desc="Inference Query"):
        predict_case(load_dir, save_dir, name, metadata_filepath, zscore_filepath, trainer, model, config, target_particle_size, target_spacing, processes, min_rel_particle_size, zscore_norm)


def predict_case(load_dir, save_dir, name, metadata_filepath, zscore_filepath, trainer, model, config, target_particle_size_in_pixel, target_spacing, processes, min_rel_particle_size, zscore_norm):
    print("Starting inference of sample: ", name)
    load_filepath = join(load_dir, "images", "{}.zarr".format(name))
    pred_softmax_filepath, pred_border_core_filepath, pred_border_core_tmp_filepath, pred_instance_filepath = setup_folder_structure(save_dir, name)

    with open(metadata_filepath) as f:
        metadata = json.load(f)

    with open(zscore_filepath) as f:
        zscore = json.load(f)
        zscore = zscore[zscore_norm]

    target_particle_size_in_mm = utils.pixel2mm(target_particle_size_in_pixel, target_spacing)
    target_patch_size_in_pixel = np.asarray(list(config['plans_per_stage'].values())[-1]['patch_size'])
    source_particle_size = metadata[name]["particle_size"]
    source_spacing = metadata[name]["spacing"]

    predict(load_filepath, pred_softmax_filepath, pred_border_core_filepath, pred_border_core_tmp_filepath, pred_instance_filepath, target_spacing, target_particle_size_in_mm, target_particle_size_in_pixel, target_patch_size_in_pixel,
            source_spacing, source_particle_size, trainer, model, processes, min_rel_particle_size, zscore)


def setup_folder_structure(save_dir, name):
    Path(join(save_dir, name)).mkdir(parents=True, exist_ok=True)
    pred_softmax_filepath = join(save_dir, name, "{}_softmax_tmp.zarr".format(name))
    pred_border_core_filepath = join(save_dir, name, "{}_border.zarr".format(name))
    pred_border_core_tmp_filepath = join(save_dir, name, "{}_border_tmp.zarr".format(name))
    pred_instance_filepath = join(save_dir, name, "{}".format(name))
    shutil.rmtree(pred_softmax_filepath, ignore_errors=True)
    shutil.rmtree(pred_border_core_filepath, ignore_errors=True)
    shutil.rmtree(pred_border_core_tmp_filepath, ignore_errors=True)
    shutil.rmtree(pred_instance_filepath, ignore_errors=True)
    return pred_softmax_filepath, pred_border_core_filepath, pred_border_core_tmp_filepath, pred_instance_filepath


def predict(load_filepath, pred_softmax_filepath, pred_border_core_filepath, pred_border_core_tmp_filepath, pred_instance_filepath, target_spacing, target_particle_size_in_mm, target_particle_size_in_pixel, target_patch_size_in_pixel,
            source_spacing, source_particle_size, trainer, model, processes, min_rel_particle_size, zscore):
    try:
        img = zarr.open(load_filepath, mode='r')
    except zarr.errors.PathNotFoundError as e:
        print("Filepath: ", load_filepath)
        raise e

    source_patch_size_in_pixel, source_chunk_size, resized_image_shape, resized_chunk_size = compute_zoom(img, source_spacing, source_particle_size, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel)
    img, crop_slices = pad_image(img, source_patch_size_in_pixel)
    source_patch_size_in_pixel, source_chunk_size, resized_image_shape, resized_chunk_size = compute_zoom(img, source_spacing, source_particle_size, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel)
    sampler, aggregator, chunked = create_sampler_and_aggregator(img, pred_border_core_filepath, source_patch_size_in_pixel, target_patch_size_in_pixel, resized_image_shape, source_chunk_size, resized_chunk_size, target_spacing)

    model.prediction_setup(aggregator, chunked, zscore)
    trainer.predict(model, dataloaders=sampler)
    border_core_resized_pred = aggregator.get_output()
    shutil.rmtree(pred_softmax_filepath, ignore_errors=True)

    instance_pred = border_core2instance_conversion(border_core_resized_pred, pred_border_core_tmp_filepath, crop_slices, img.shape, source_spacing, pred_instance_filepath, processes=processes)
    instance_pred = filter_small_particles(instance_pred, min_rel_particle_size)
    save_prediction(instance_pred, pred_instance_filepath, source_spacing)

    shutil.rmtree(pred_border_core_filepath, ignore_errors=True)
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
    return source_patch_size_in_pixel, source_chunk_size, resized_image_shape, resized_chunk_size


def create_sampler_and_aggregator(img, pred_border_core_filepath, source_patch_size_in_pixel, target_patch_size_in_pixel, resized_image_shape, source_chunk_size, resized_chunk_size, target_spacing):
    region_class_order = None
    batch_size = 6
    num_workers = 12
    num_channels = 3
    if np.prod(resized_image_shape) < 1000*1000*500:
        pred = zarr.open(pred_border_core_filepath, mode='w', shape=(num_channels, *resized_image_shape), chunks=(3, 64, 64, 64), dtype=np.float32)
        blosc.set_nthreads(4)
        sampler = GridSampler(img, image_size=img.shape[-3:], patch_size=source_patch_size_in_pixel, patch_overlap=source_patch_size_in_pixel // 2)
        if not np.array_equal(img.shape, resized_image_shape):
            sampler = ResizeSampler(sampler, target_size=target_patch_size_in_pixel, image_size=resized_image_shape[-3:], patch_size=target_patch_size_in_pixel, patch_overlap=target_patch_size_in_pixel // 2)
        sampler = SamplerDataset(sampler)
        sampler = DataLoader(sampler, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)
        aggregator = WeightedSoftmaxAggregator(pred, image_size=resized_image_shape[-3:], patch_size=target_patch_size_in_pixel, region_class_order=region_class_order)
        chunked = False
    else:
        pred = zarr.open(pred_border_core_filepath, mode='w', shape=resized_image_shape[-3:], chunks=(64, 64, 64), dtype=np.uint8)
        blosc.set_nthreads(4)
        sampler = ChunkedGridSampler(img, image_size=img.shape[-3:], patch_size=source_patch_size_in_pixel, patch_overlap=source_patch_size_in_pixel // 2, chunk_size=source_chunk_size)
        if not np.array_equal(img.shape, resized_image_shape):
            sampler = ChunkedResizeSampler(sampler, target_size=target_patch_size_in_pixel, image_size=resized_image_shape[-3:], patch_size=target_patch_size_in_pixel, patch_overlap=target_patch_size_in_pixel // 2, chunk_size=resized_chunk_size)
        sampler = SamplerDataset(sampler)
        sampler = DataLoader(sampler, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=False)
        aggregator = ResizeChunkedWeightedSoftmaxAggregator(pred, image_size=resized_image_shape[-3:], patch_size=target_patch_size_in_pixel, patch_overlap=target_patch_size_in_pixel // 2, chunk_size=resized_chunk_size, spacing=target_spacing, region_class_order=region_class_order)
        chunked = True
    return sampler, aggregator, chunked


def border_core2instance_conversion(border_core_pred, pred_border_core_tmp_filepath, crop_slices, original_size,
                                    source_spacing, save_filepath, debug=False, dtype=np.uint16, processes=None):
    if debug:
        border_core_pred_resampled = np.array(border_core_pred)
        utils.save_nifti(save_filepath + "_border_core_zoomed.nii.gz", border_core_pred_resampled, source_spacing)
    instance_pred, num_instances = border_core2instance(border_core_pred, pred_border_core_tmp_filepath, processes, dtype=dtype, progressbar=False)
    if debug:
        utils.save_nifti(save_filepath + "_zoomed.nii.gz", instance_pred, source_spacing)
    instance_pred = ski_transform.resize(instance_pred, original_size, 0, mode="edge", anti_aliasing=False)
    instance_pred = crop_pred(instance_pred, crop_slices)
    return instance_pred


def filter_small_particles(instance_pred, min_rel_particle_size):
    if min_rel_particle_size is None:
        return instance_pred

    particle_voxels = cc3d.statistics(instance_pred)["voxel_counts"]
    particle_voxels = particle_voxels[1:]  # Remove background from list

    mean_particle_voxels = np.mean(particle_voxels)
    min_threshold = min_rel_particle_size * mean_particle_voxels

    instances_to_remove = np.arange(1, len(particle_voxels) + 1, dtype=int)
    instances_to_remove = instances_to_remove[particle_voxels < min_threshold]

    if len(instances_to_remove) > 0:
        target_values = np.zeros_like(instances_to_remove, dtype=int)
        shape = instance_pred.shape
        instance_pred = npi.remap(instance_pred.flatten(), instances_to_remove, target_values)
        instance_pred = instance_pred.reshape(shape)

    return instance_pred


def save_prediction(instance_pred, save_filepath, source_spacing):
    instance_pred_zarr = zarr.open(save_filepath + ".zarr", shape=instance_pred.shape, mode='w')
    instance_pred_zarr[...] = instance_pred
    instance_pred_zarr.attrs["spacing"] = source_spacing


def pad_image(image, target_image_shape):
    if np.any(image.shape < target_image_shape):
        pad_kwargs = {'constant_values': 0}
        image = np.asarray(image)
        image, slices = pad_nd_image(image, target_image_shape, "constant", pad_kwargs, True, None)
        return image, slices
    else:
        return image, None


def crop_pred(pred, crop_slices):
    if crop_slices is not None:
        pred = pred[tuple(crop_slices)]
    return pred


def compute_patch_size(target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel, source_spacing, source_particle_size_in_mm, image_shape):
    size_conversion_factor = utils.compute_size_conversion_factor(source_particle_size_in_mm, source_spacing, target_particle_size_in_mm, target_spacing)
    size_conversion_factor = np.around(size_conversion_factor, decimals=3)
    source_patch_size_in_pixel = np.rint(target_patch_size_in_pixel * size_conversion_factor).astype(int)
    return target_patch_size_in_pixel, source_patch_size_in_pixel, size_conversion_factor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the base folder that contains the dataset structured in the form of the directories 'images' and 'instance_seg' and the files metadata.json and zscore.json.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the save folder.")
    parser.add_argument('-m', "--model", required=True, help="Absolute path to the model directory.")
    parser.add_argument('-n', "--name", required=False, type=str, nargs="+", help="The name(s) without extension of the image(s) that should be used for inference. Multiple names must be separated by spaces.")
    parser.add_argument('-t', "--task", required=False, default=310, type=int, help="The task ID.")
    parser.add_argument('-target_particle_size', default=(60, 60, 60), required=False, type=float, nargs=3,
                        help="The target particle size in pixels given as three numbers separate by spaces.")
    parser.add_argument('-target_spacing', default=(0.1, 0.1, 0.1), required=False, type=float, nargs=3,
                        help="The target spacing in millimeters given as three numbers separate by spaces.")
    parser.add_argument('-f', "--fold", required=False, default=(0, 1, 2, 3, 4), type=int, nargs="+", help="The folds to use. 0, 1, 2, 3, 4 or a combination.")
    parser.add_argument('-p', '--processes', required=False, default=12, type=int, help="Number of processes to use for parallel processing. None to disable multiprocessing.")
    parser.add_argument("-min_rel_particle_size", required=False, default=0.005, type=float, help="Minimum relative particle size used for filtering.")
    parser.add_argument('-zscore_norm', required=False, default="global_zscore", type=str,
                        help="(Optional) The type of normalization to use. Either 'global_zscore' or 'local_zscore'.")
    args = parser.parse_args()

    print("Names: ", args.name)

    trainer, model, config = setup_model(args.model, args.fold)
    predict_cases(args.input, args.output, args.name, trainer, model, config, args.target_particle_size, args.target_spacing, args.processes, args.min_rel_particle_size, args.zscore_norm)
