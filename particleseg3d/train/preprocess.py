from pathlib import Path
import numpy as np
from tqdm import tqdm
from particleseg3d.utils import utils
import json
from os.path import join
from particleseg3d.train.instance2border_core import instance2border_core
import zarr
from acvl_utils.miscellaneous.ptqdm import ptqdm
import pickle
from skimage.measure import regionprops
import argparse
from typing import List, Tuple, Dict, Any


def preprocess_all(load_dir: str, names: List[str], save_dir: str, target_spacing: float,
                   target_particle_size_in_pixel: int, dataset_name: str, processes: int,
                   border_thickness_in_pixel: int, gpu: bool, zscore: Tuple[float, float]) -> None:
    """
    Preprocesses all the samples in the dataset.

    :param load_dir: Path to the base directory that contains the dataset structured in the form of the directories 'images' and 'instance_seg' and the files metadata.json and zscore.json.
    :param names: The name(s) without extension of the image(s) that should be used for training.
    :param save_dir: Path to the preprocessed dataset directory.
    :param target_spacing: The target spacing in millimeters given as three numbers separate by spaces.
    :param target_particle_size_in_pixel: The target particle size in pixels given as three numbers separate by spaces.
    :param dataset_name: The name of the preprocessed dataset.
    :param processes: Number of processes to use for parallel processing. None to disable multiprocessing.
    :param border_thickness_in_pixel: Border thickness in pixel.
    :param gpu: Flag indicating whether to use the GPU for preprocessing.
    :param zscore: The z-score used for intensity normalization.
    """
    metadata_load_filepath = join(load_dir, "metadata.json")

    with open(metadata_load_filepath) as f:
        metadata = json.load(f)

    target_spacing = [target_spacing] * 3
    target_particle_size_in_pixel = [target_particle_size_in_pixel] * 3

    image_save_dir = join(save_dir, dataset_name, "imagesTr")
    semantic_seg_save_dir = join(save_dir, dataset_name, "labelsTr")
    instance_seg_save_dir = join(save_dir, dataset_name, "labelsTr_instance")
    semantic_seg_zarr_save_dir = join(save_dir, dataset_name, "labelsTr_zarr")
    instance_seg_zarr_save_dir = join(save_dir, dataset_name, "labelsTr_instance_zarr")
    Path(instance_seg_save_dir).mkdir(parents=True, exist_ok=True)
    Path(image_save_dir).mkdir(parents=True, exist_ok=True)
    Path(semantic_seg_save_dir).mkdir(parents=True, exist_ok=True)

    for name in names:
        if name not in metadata:
            raise RuntimeError("{} is missing in metadata!".format(name))

    image_load_filepaths = [join(load_dir, "images", name + ".nii.gz") for name in names]
    seg_load_filepaths = [join(load_dir, "instance_seg", name + ".nii.gz") for name in names]

    if processes is None:
        for i in tqdm(range(len(names))):
            preprocess_single(i, names=names, image_load_filepaths=image_load_filepaths, seg_load_filepaths=seg_load_filepaths, metadata_load_filepath=metadata_load_filepath,
                      image_save_dir=image_save_dir, semantic_seg_save_dir=semantic_seg_save_dir, instance_seg_save_dir=instance_seg_save_dir,
                      semantic_seg_zarr_save_dir=semantic_seg_zarr_save_dir, instance_seg_zarr_save_dir=instance_seg_zarr_save_dir, target_spacing=target_spacing,
                      target_particle_size_in_pixel=target_particle_size_in_pixel,
                      border_thickness_in_pixel=border_thickness_in_pixel, gpu=gpu, zscore=zscore)
    else:
        ptqdm(preprocess_single, range(len(names)), processes, names=names, image_load_filepaths=image_load_filepaths, seg_load_filepaths=seg_load_filepaths, metadata_load_filepath=metadata_load_filepath,
                  image_save_dir=image_save_dir, semantic_seg_save_dir=semantic_seg_save_dir, instance_seg_save_dir=instance_seg_save_dir,
                  semantic_seg_zarr_save_dir=semantic_seg_zarr_save_dir, instance_seg_zarr_save_dir=instance_seg_zarr_save_dir, target_spacing=target_spacing,
                  target_particle_size_in_pixel=target_particle_size_in_pixel,
                  border_thickness_in_pixel=border_thickness_in_pixel, gpu=gpu, zscore=zscore)

    utils.generate_dataset_json(join(save_dir, dataset_name, 'dataset.json'), join(save_dir, dataset_name, "imagesTr"), None, ("noNorm",), {0: 'bg', 1: 'core', 2: 'border'}, dataset_name)

    gen_regionprops(join(save_dir, dataset_name, "labelsTr_instance"), join(save_dir, dataset_name, "regionprops.pkl"))


def preprocess_single(i: int,
                      names: List[str],
                      image_load_filepaths: List[str],
                      seg_load_filepaths: List[str],
                      metadata_load_filepath: str,
                      image_save_dir: str,
                      semantic_seg_save_dir: str,
                      instance_seg_save_dir: str,
                      semantic_seg_zarr_save_dir: str,
                      instance_seg_zarr_save_dir: str,
                      target_spacing: List[float],
                      target_particle_size_in_pixel: List[int],
                      border_thickness_in_pixel: int,
                      gpu: bool,
                      zscore: Tuple[float, float]) -> None:
    """
    Preprocess a single 3D particle segmentation image.

    Args:
        i (int): Index of the image to preprocess.
        names (List[str]): Names of the images to preprocess.
        image_load_filepaths (List[str]): Paths to the input 3D particle segmentation image files.
        seg_load_filepaths (List[str]): Paths to the input instance segmentation image files.
        metadata_load_filepath (str): Path to the metadata file.
        image_save_dir (str): Path to the directory to save the preprocessed images.
        semantic_seg_save_dir (str): Path to the directory to save the semantic segmentation images.
        instance_seg_save_dir (str): Path to the directory to save the instance segmentation images.
        semantic_seg_zarr_save_dir (str): Path to the directory to save the semantic segmentation images in zarr format.
        instance_seg_zarr_save_dir (str): Path to the directory to save the instance segmentation images in zarr format.
        target_spacing (List[float]): Target spacing in millimeters.
        target_particle_size_in_pixel (List[int]): Target particle size in pixels.
        border_thickness_in_pixel (int): Border thickness in pixels.
        gpu (bool): If True, use GPU for resampling.
        zscore: The z-score used for intensity normalization.

    Returns:
        None
    """
    name = names[i]
    image_load_filepath = image_load_filepaths[i]
    seg_load_filepath = seg_load_filepaths[i]

    with open(metadata_load_filepath) as f:
        metadata = json.load(f)

    image = utils.load_nifti(image_load_filepath)
    instance_seg = utils.load_nifti(seg_load_filepath)

    zscore = {"mean": zscore[0], "std": zscore[1]}
    image = utils.standardize(image, zscore)

    image_shape = image.shape
    source_particle_size_in_mm = [metadata[name]["particle_size"]] * 3
    source_spacing = [metadata[name]["spacing"]] * 3
    target_particle_size_in_mm = tuple(utils.pixel2mm(target_particle_size_in_pixel, target_spacing))

    size_conversion_factor = utils.compute_size_conversion_factor(source_particle_size_in_mm, source_spacing, target_particle_size_in_mm, target_spacing)
    target_patch_size_in_pixel = np.rint(np.asarray(image_shape) / size_conversion_factor).astype(int)
    target_patch_size_in_pixel = target_patch_size_in_pixel.tolist()

    image = utils.resample(image, target_patch_size_in_pixel, gpu=gpu, disable=True)
    instance_seg = utils.resample(instance_seg, target_patch_size_in_pixel, gpu=gpu, seg=True, disable=True)
    patch_name = "{}".format(name)
    image_save_filepath = join(image_save_dir, patch_name + "_0000.nii.gz")
    utils.save_nifti(image_save_filepath, image, spacing=target_spacing)
    semantic_seg = instance2border_core(instance_seg, border_thickness_in_pixel)
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


def gen_regionprops(load_dir: str, metadata_filepath: str) -> None:
    """Extracts regionprops features from the given instance segmentation data and saves the data to the given file.

    Args:
        load_dir (str): Absolute path to the directory containing instance segmentation data.
        metadata_filepath (str): Absolute path to the file where the extracted regionprops features should be saved.
    """
    names = utils.load_filepaths(load_dir, return_path=False, return_extension=False)
    metadata = {}

    len_props_total, len_props_filtered_total = 0, 0
    for name in tqdm(names):
        instance_seg = utils.load_nifti(join(load_dir, name + ".nii.gz"))
        props, len_props, len_props_filtered = gen_regionprops_single(instance_seg)
        len_props_total += len_props
        len_props_filtered_total += len_props_filtered
        metadata[name] = props

    with open(metadata_filepath, 'wb') as handle:
        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)


def gen_regionprops_single(instance_seg: np.ndarray) -> Tuple[Dict[int, Tuple[int, int, int, int, int, int]], int, int]:
    """Extracts regionprops features from a single instance segmentation volume.

    Args:
        instance_seg (np.ndarray): A 3D numpy array containing the instance segmentation data.

    Returns:
        A tuple containing the extracted regionprops features, the number of total regionprops, and the number of filtered regionprops.
    """
    props = {prop.label: prop.bbox for prop in regionprops(instance_seg)}

    props_filtered = {}
    image_shape = instance_seg.shape
    for label, bbox in props.items():
        bbox_reshaped = [[bbox[i], bbox[i + len(bbox) // 2]] for i in range(len(bbox) // 2)]
        bbox_reshaped = np.asarray(bbox_reshaped)
        ok = True
        for axis in range(len(image_shape)):
            if bbox_reshaped[axis][0] == 0 or bbox_reshaped[axis][1] == image_shape[axis]:
                ok = False
        if ok:
            props_filtered[label] = bbox

    return props_filtered, len(props), len(props_filtered)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', "--input", required=True,
                        help="Absolute input path to the base folder that contains the dataset structured in the form of the directories 'images' and 'instance_seg' and the file metadata.json.")
    parser.add_argument('-o', "--output", required=True, help="Absolute output path to the preprocessed dataset directory.")
    parser.add_argument('-n', "--name", required=False, type=str, default=None, nargs="+", help="(Optional) The name(s) without extension of the image(s) that should be used for training. Multiple names must be separated by spaces.")
    parser.add_argument('-t', '--task', required=False, default=500, type=int, help="(Optional) The task id that should be assigned to this dataset.")
    parser.add_argument('-z', '--zscore', default=(5850.29762143569, 7078.294543817302), required=False, type=float, nargs=2,
                        help="(Optional) The target spacing in millimeters given as three numbers separate by spaces.")
    parser.add_argument('-target_particle_size', default=60, required=False, type=int,
                        help="(Optional) The target particle size in pixels given as three numbers separate by spaces.")
    parser.add_argument('-target_spacing', default=0.1, required=False, type=float,
                        help="(Optional) The target spacing in millimeters given as three numbers separate by spaces.")
    parser.add_argument('-p', '--processes', required=False, default=None, type=int, help="(Optional) Number of processes to use for parallel processing. None to disable multiprocessing.")
    parser.add_argument('-thickness', required=False, default=3, type=int, help="(Optional) Border thickness in pixel.")
    parser.add_argument('--disable_gpu', required=False, default=False, action="store_true", help="(Optional) Disables use of the GPU for preprocessing.")
    args = parser.parse_args()

    parser = argparse.ArgumentParser(description="Preprocess a dataset for training a particle segmentation model.")

    names = args.name

    if names is None:
        names = utils.load_filepaths(join(args.input, "images"), extension=".nii.gz", return_path=False, return_extension=False)

    print("Samples: ", names)
    print("Num samples: ", len(names))

    dataset_name = "Task{}_ParticleSeg3D".format(str(args.task).zfill(3))

    preprocess_all(args.input, names, args.output, args.target_spacing, args.target_particle_size, dataset_name, args.processes, args.thickness, not args.disable_gpu, args.zscore)


if __name__ == '__main__':
    main()
