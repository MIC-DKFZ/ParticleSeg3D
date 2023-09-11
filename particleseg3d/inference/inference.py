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
from particleseg3d.inference.model_nnunet import Nnunet
import json
from particleseg3d.inference.border_core2instance import border_core2instance
from skimage import transform as ski_transform
from pathlib import Path
import argparse
import pickle
from batchgenerators.augmentations.utils import pad_nd_image
import cc3d
import numpy_indexed as npi
from typing import List, Tuple, Any, Optional, Dict, Type
from skimage import transform as ski_transform


def setup_model(model_dir: str, folds: List[int], trainer: str = "nnUNetTrainerV2_slimDA5_touchV5__nnUNetPlansv2.1") -> Tuple[pl.Trainer, Nnunet, Dict[str, Any]]:
    """
    Set up the model for inference.

    Args:
        model_dir: The directory containing the model files.
        folds: The folds to use for inference.
        trainer: The name of the trainer.

    Returns:
        A tuple containing:
            trainer: The PyTorch Lightning Trainer object.
            model: The initialized Nnunet model.
            config: The model configuration.
    """
    with open(join(model_dir, trainer, "plans.pkl"), 'rb') as handle:
        config = pickle.load(handle)

    model = Nnunet(join(model_dir, trainer), folds=folds, nnunet_trainer=trainer, configuration="3d_fullres")
    model.eval()
    trainer = pl.Trainer(gpus=1, precision=16, logger=False)
    return trainer, model, config


def predict_cases(
    load_dir: str,
    save_dir: str,
    names: Optional[List[str]],
    trainer: pl.Trainer,
    model: Nnunet,
    config: Dict[str, Any],
    target_particle_size: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    batch_size: int,
    processes: int,
    min_rel_particle_size: float,
    zscore: Tuple[float, float]
) -> None:
    """
    Perform inference on multiple cases.

    Args:
        load_dir: The directory containing the input data.
        save_dir: The directory to save the output predictions.
        names: Optional. The names of the cases to process. If None, all cases in `load_dir` will be processed.
        trainer: The PyTorch Lightning Trainer object.
        model: The initialized Nnunet model.
        config: The model configuration.
        target_particle_size: The target particle size in pixels.
        target_spacing: The target spacing in millimeters.
        batch_size: The batch size to use during inference.
        processes: The number of processes to use for parallel processing.
        min_rel_particle_size: The minimum relative particle size used for filtering.
        zscore: The z-score used for intensity normalization.
    """
    image_dir = join(load_dir, "images")
    metadata_filepath = join(load_dir, "metadata.json")

    target_particle_size = [target_particle_size] * 3
    target_spacing = [target_spacing] * 3

    if names is None:
        names = utils.load_filepaths(image_dir, return_path=False, return_extension=False)

    print("Samples: ", names)

    for name in tqdm(names, desc="Inference Query"):
        predict_case(image_dir, save_dir, name, metadata_filepath, zscore, trainer, model, config, target_particle_size, target_spacing, processes, min_rel_particle_size, batch_size)


def predict_case(
    load_dir: str,
    save_dir: str,
    name: str,
    metadata_filepath: str,
    zscore: Tuple[float, float],
    trainer: pl.Trainer,
    model: Nnunet,
    config: Dict[str, Any],
    target_particle_size_in_pixel: float,
    target_spacing: Tuple[float, float, float],
    processes: int,
    min_rel_particle_size: float,
    batch_size: int
) -> None:
    """
    Perform inference on a single case.

    Args:
        load_dir: The directory containing the input data.
        save_dir: The directory to save the output predictions.
        name: The name of the case to process.
        metadata_filepath: The file path to the metadata.
        zscore: The z-score used for intensity normalization.
        trainer: The PyTorch Lightning Trainer object.
        model: The initialized Nnunet model.
        config: The model configuration.
        target_particle_size_in_pixel: The target particle size in pixels.
        target_spacing: The target spacing in millimeters.
        processes: The number of processes to use for parallel processing.
        min_rel_particle_size: The minimum relative particle size used for filtering.
        batch_size: The batch size to use during inference.
    """
    print("Starting inference of sample: ", name)
    load_filepath = join(load_dir, "{}.zarr".format(name))
    pred_softmax_filepath, pred_border_core_filepath, pred_border_core_tmp_filepath, pred_instance_filepath = setup_folder_structure(save_dir, name)

    with open(metadata_filepath) as f:
        metadata = json.load(f)

    zscore = {"mean": zscore[0], "std": zscore[1]}

    target_particle_size_in_mm = utils.pixel2mm(target_particle_size_in_pixel, target_spacing)
    target_patch_size_in_pixel = np.asarray(list(config['plans_per_stage'].values())[-1]['patch_size'])
    source_particle_size = [metadata[name]["particle_size"]] * 3
    source_spacing = [metadata[name]["spacing"]] * 3

    predict(load_filepath, pred_softmax_filepath, pred_border_core_filepath, pred_border_core_tmp_filepath, pred_instance_filepath, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel,
            source_spacing, source_particle_size, trainer, model, processes, min_rel_particle_size, zscore, batch_size)


def setup_folder_structure(
    save_dir: str,
    name: str
) -> Tuple[str, str, str, str]:
    """
    Set up the folder structure for saving predictions.

    Args:
        save_dir: The directory to save the predictions.
        name: The name of the case.

    Returns:
        pred_softmax_filepath: The file path for predicted softmax.
        pred_border_core_filepath: The file path for predicted border and core.
        pred_border_core_tmp_filepath: The temporary file path for predicted border and core.
        pred_instance_filepath: The file path for predicted instances.
    """
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


def predict(
    load_filepath: str,
    pred_softmax_filepath: str,
    pred_border_core_filepath: str,
    pred_border_core_tmp_filepath: str,
    pred_instance_filepath: str,
    target_spacing: Tuple[float, float, float],
    target_particle_size_in_mm: Tuple[float, float, float],
    target_patch_size_in_pixel: float,
    source_spacing: Tuple[float, float, float],
    source_particle_size: Tuple[float, float, float],
    trainer: pl.Trainer,
    model: Nnunet,
    processes: int,
    min_rel_particle_size: float,
    zscore: Dict[str, Any],
    batch_size: int
) -> None:
    """
    Perform the prediction for a single case.

    Args:
        load_filepath: The file path of the input data.
        pred_softmax_filepath: The file path to save the predicted softmax.
        pred_border_core_filepath: The file path to save the predicted border and core.
        pred_border_core_tmp_filepath: The temporary file path to save the predicted border and core.
        pred_instance_filepath: The file path to save the predicted instances.
        target_spacing: The target spacing in millimeters.
        target_particle_size_in_mm: The target particle size in millimeters.
        target_patch_size_in_pixel: The target patch size in pixels.
        source_spacing: The source spacing in millimeters.
        source_particle_size: The source particle size in millimeters.
        trainer: The PyTorch Lightning Trainer object.
        model: The initialized Nnunet model.
        processes: The number of processes to use for parallel processing.
        min_rel_particle_size: The minimum relative particle size used for filtering.
        zscore: The z-score normalization values.
        batch_size: The batch size to use during inference.
    """
    try:
        img = zarr.open(load_filepath, mode='r')
    except zarr.errors.PathNotFoundError as e:
        print("Filepath: ", load_filepath)
        raise e

    source_patch_size_in_pixel, source_chunk_size, resized_image_shape, resized_chunk_size = compute_zoom(img, source_spacing, source_particle_size, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel)
    img, crop_slices = pad_image(img, source_patch_size_in_pixel)
    source_patch_size_in_pixel, source_chunk_size, resized_image_shape, resized_chunk_size = compute_zoom(img, source_spacing, source_particle_size, target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel)
    sampler, aggregator, chunked = create_sampler_and_aggregator(img, pred_border_core_filepath, source_patch_size_in_pixel, target_patch_size_in_pixel, resized_image_shape, source_chunk_size, resized_chunk_size, target_spacing, batch_size, processes)

    model.prediction_setup(aggregator, chunked, zscore)
    trainer.predict(model, dataloaders=sampler)
    border_core_resized_pred = aggregator.get_output()
    shutil.rmtree(pred_softmax_filepath, ignore_errors=True)

    instance_pred = border_core2instance_conversion(border_core_resized_pred, pred_border_core_tmp_filepath, crop_slices, img.shape, source_spacing, pred_instance_filepath, processes=processes)
    instance_pred = filter_small_particles(instance_pred, min_rel_particle_size)
    save_prediction(instance_pred, pred_instance_filepath, source_spacing)

    shutil.rmtree(pred_border_core_filepath, ignore_errors=True)
    shutil.rmtree(pred_border_core_tmp_filepath, ignore_errors=True)


def compute_zoom(
    img: Any,
    source_spacing: Tuple[float, float, float],
    source_particle_size: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    target_particle_size_in_mm: Tuple[float, float, float],
    target_patch_size_in_pixel: float
) -> Tuple[float, float, Any, Any]:
    """
    Compute the zoom parameters for resizing.

    Args:
        img: The input image.
        source_spacing: The source spacing in millimeters.
        source_particle_size: The source particle size in millimeters.
        target_spacing: The target spacing in millimeters.
        target_particle_size_in_mm: The target particle size in millimeters.
        target_patch_size_in_pixel: The target patch size in pixels.

    Returns:
        source_patch_size_in_pixel: The source patch size in pixels.
        source_chunk_size: The source chunk size.
        resized_image_shape: The shape of the resized image.
        resized_chunk_size: The resized chunk size.
    """
    if np.array_equal(target_particle_size_in_mm, [0, 0, 0]):
        return target_patch_size_in_pixel, target_patch_size_in_pixel * 4, img.shape, target_patch_size_in_pixel * 4
    image_shape = np.asarray(img.shape[-3:])
    source_particle_size_in_mm = tuple(source_particle_size)
    source_spacing = tuple(source_spacing)
    _, source_patch_size_in_pixel, size_conversion_factor = compute_patch_size(target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel, source_spacing, source_particle_size_in_mm)
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


def create_sampler_and_aggregator(
    img: Any,
    pred_border_core_filepath: str,
    source_patch_size_in_pixel: float,
    target_patch_size_in_pixel: float,
    resized_image_shape: Any,
    source_chunk_size: Any,
    resized_chunk_size: Any,
    target_spacing: Tuple[float, float, float],
    batch_size: int,
    num_workers: int
) -> Tuple[Any, Any, bool]:
    """
    Create the sampler and aggregator for prediction.

    Args:
        img: The input image.
        pred_border_core_filepath: The file path to save the predicted border and core.
        source_patch_size_in_pixel: The source patch size in pixels.
        target_patch_size_in_pixel: The target patch size in pixels.
        resized_image_shape: The shape of the resized image.
        source_chunk_size: The source chunk size.
        resized_chunk_size: The resized chunk size.
        target_spacing: The target spacing in millimeters.
        batch_size: The batch size to use during inference.

    Returns:
        sampler: The data sampler.
        aggregator: The aggregator for prediction.
        chunked: A flag indicating if chunked processing is used.
    """
    region_class_order = None
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


def border_core2instance_conversion(
    border_core_pred: Any,
    pred_border_core_tmp_filepath: str,
    crop_slices: Any,
    original_size: Any,
    source_spacing: Tuple[float, float, float],
    save_filepath: str,
    debug: bool = False,
    dtype: Type = np.uint16,
    processes: Optional[int] = None
) -> Any:
    """
    Convert border and core predictions to instance predictions.

    Args:
        border_core_pred: The predicted border and core.
        pred_border_core_tmp_filepath: The temporary file path for predicted border and core.
        crop_slices: The crop slices for the original image.
        original_size: The original size of the image.
        source_spacing: The source spacing in millimeters.
        save_filepath: The file path to save the predicted instances.
        debug: Optional. Enable debug mode.
        dtype: The data type of the output instance predictions.
        processes: Optional. The number of processes to use for parallel processing.

    Returns:
        instance_pred: The predicted instances.
    """
    if debug:
        border_core_pred_resampled = np.array(border_core_pred)
        utils.save_nifti(save_filepath + "_border_core_zoomed.nii.gz", border_core_pred_resampled, source_spacing)
    instance_pred, num_instances = border_core2instance(border_core_pred, pred_border_core_tmp_filepath, processes, dtype=dtype, progressbar=False)
    if debug:
        utils.save_nifti(save_filepath + "_zoomed.nii.gz", instance_pred, source_spacing)
    instance_pred = ski_transform.resize(instance_pred, original_size, 0, mode="edge", anti_aliasing=False)
    instance_pred = crop_pred(instance_pred, crop_slices)
    return instance_pred


def filter_small_particles(
    instance_pred: Any,
    min_rel_particle_size: Optional[float]
) -> Any:
    """
    Filter out small particles from the instance predictions.

    Args:
        instance_pred: The predicted instances.
        min_rel_particle_size: Optional. The minimum relative particle size used for filtering.

    Returns:
        instance_pred: The filtered instance predictions.
    """
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


def save_prediction(
    instance_pred: Any,
    save_filepath: str,
    source_spacing: Tuple[float, float, float]
) -> None:
    """
    Save the predicted instances to a file.

    Args:
        instance_pred: The predicted instances.
        save_filepath: The file path to save the instances.
        source_spacing: The source spacing in millimeters.

    Returns:
        None
    """
    instance_pred_zarr = zarr.open(save_filepath + ".zarr", shape=instance_pred.shape, mode='w')
    instance_pred_zarr[...] = instance_pred
    instance_pred_zarr.attrs["spacing"] = source_spacing


def pad_image(
    image: Any,
    target_image_shape: Any
) -> Tuple[Any, Optional[Any]]:
    """
    Pad the image to match the target image shape.

    Args:
        image: The input image.
        target_image_shape: The target image shape.

    Returns:
        image: The padded image.
        slices: The crop slices if padding is applied, None otherwise.
    """
    if np.any(image.shape < target_image_shape):
        pad_kwargs = {'constant_values': 0}
        image = np.asarray(image)
        image, slices = pad_nd_image(image, target_image_shape, "constant", pad_kwargs, True, None)
        return image, slices
    else:
        return image, None


def crop_pred(
    pred: Any,
    crop_slices: Optional[Any]
) -> Any:
    """
    Crop the prediction using the crop slices.

    Args:
        pred: The prediction.
        crop_slices: The crop slices.

    Returns:
        cropped_pred: The cropped prediction.
    """
    if crop_slices is not None:
        pred = pred[tuple(crop_slices)]
    return pred


def compute_patch_size(
    target_spacing: Tuple[float, float, float],
    target_particle_size_in_mm: Tuple[float, float, float],
    target_patch_size_in_pixel: float,
    source_spacing: Tuple[float, float, float],
    source_particle_size_in_mm: Tuple[float, float, float]
) -> Tuple[float, float, Any]:
    """
    Compute the patch size for resizing.

    Args:
        target_spacing: The target spacing in millimeters.
        target_particle_size_in_mm: The target particle size in millimeters.
        target_patch_size_in_pixel: The target patch size in pixels.
        source_spacing: The source spacing in millimeters.
        source_particle_size_in_mm: The source particle size in millimeters.
        image_shape: The shape of the image.

    Returns:
        target_patch_size_in_pixel: The target patch size in pixels.
        source_patch_size_in_pixel: The source patch size in pixels.
        size_conversion_factor: The size conversion factor.
    """
    size_conversion_factor = utils.compute_size_conversion_factor(source_particle_size_in_mm, source_spacing, target_particle_size_in_mm, target_spacing)
    size_conversion_factor = np.around(size_conversion_factor, decimals=3)
    source_patch_size_in_pixel = np.rint(target_patch_size_in_pixel * size_conversion_factor).astype(int)
    return target_patch_size_in_pixel, source_patch_size_in_pixel, size_conversion_factor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the base folder that contains the dataset structured in the form of the directories 'images' and the metadata.json.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the save folder.")
    parser.add_argument('-m', "--model", required=True, help="Absolute path to the model directory. Example: /path/to/model/Task310_particle_seg")
    parser.add_argument('-n', "--name", required=False, type=str, nargs="+", help="(Optional) The name(s) without extension of the image(s) that should be used for inference. Multiple names must be separated by spaces.")
    parser.add_argument('-z', '--zscore', default=(5850.29762143569, 7078.294543817302), required=False, type=float, nargs=2,
                        help="(Optional) The target spacing in millimeters given as three numbers separate by spaces.")
    parser.add_argument('-target_particle_size', default=60, required=False, type=int,
                        help="(Optional) The target particle size in pixels given as three numbers separate by spaces.")
    parser.add_argument('-target_spacing', default=0.1, required=False, type=float,
                        help="(Optional) The target spacing in millimeters given as three numbers separate by spaces.")
    parser.add_argument('-f', "--fold", required=False, default=(0, 1, 2, 3, 4), type=int, nargs="+", help="(Optional) The folds to use. 0, 1, 2, 3, 4 or a combination.")
    parser.add_argument('-batch_size', default=6, required=False, type=int,
                        help="(Optional) The batch size to use during each inference iteration. A higher batch size decreases inference time, but increases the required GPU memory.")
    parser.add_argument('-p', '--processes', required=False, default=4, type=int, help="(Optional) Number of processes to use for parallel processing. Zero to disable multiprocessing.")
    parser.add_argument("-min_rel_particle_size", required=False, default=0.005, type=float, help="(Optional) Minimum relative particle size used for filtering.")
    args = parser.parse_args()

    trainer, model, config = setup_model(args.model, args.fold)
    predict_cases(args.input, args.output, args.name, trainer, model, config, args.target_particle_size, args.target_spacing, args.batch_size, args.processes, args.min_rel_particle_size, args.zscore)


if __name__ == '__main__':
    main()
