import numpy as np
import SimpleITK as sitk
import torch
from torch.nn import functional
from tqdm import tqdm
from os.path import join
from natsort import natsorted
import os
from acvl_utils.miscellaneous.ptqdm import ptqdm
import json
from typing import Tuple, List, Union


def resample(image: np.ndarray, target_shape: Tuple[int], seg: bool = False, gpu: bool = True,
             smooth_seg: bool = True, processes: int = None, desc: str = None, disable: bool = True) -> np.ndarray:
    """
    Resample an image to a target shape.

    Args:
        image (np.ndarray): The image to resample.
        target_shape (Tuple[int]): The shape to resample to.
        seg (bool, optional): Whether the image is a segmentation. Defaults to False.
        gpu (bool, optional): Whether to use the GPU. Defaults to True.
        smooth_seg (bool, optional): Whether to smooth the segmentation. Defaults to True.
        processes (int, optional): The number of processes to use. Defaults to None.
        desc (str, optional): A description of the progress bar. Defaults to None.
        disable (bool, optional): Whether to disable the progress bar. Defaults to True.

    Returns:
        np.ndarray: The resampled image.
    """
    if all([i == j for i, j in zip(image.shape[1:], target_shape)]):
        return image

    image = torch.from_numpy(image[np.newaxis, np.newaxis, ...].astype(np.float32))
    with torch.no_grad():
        if gpu:
            image = image.cuda()
        if not seg:
            image = functional.interpolate(image, target_shape, mode='trilinear')
        else:
            if not smooth_seg:
                image = functional.interpolate(image, target_shape, mode='nearest')
            else:
                image = resample_seg_smooth(image, target_shape, processes, desc, disable)

    image = image.cpu().numpy()[0][0]
    torch.cuda.empty_cache()
    return image


def resample_seg_smooth(seg: torch.Tensor, target_shape: Tuple[int], processes: int, desc: str,
                        disable: bool) -> torch.Tensor:
    """
    Smoothly resample a segmentation.

    Args:
        seg (torch.Tensor): The segmentation to resample.
        target_shape (Tuple[int]): The shape to resample to.
        processes (int): The number of processes to use.
        desc (str): A description of the progress bar.
        disable (bool): Whether to disable the progress bar.

    Returns:
        torch.Tensor: The resampled segmentation.
    """
    target_shape_tmp = np.asarray(seg.shape)
    target_shape_tmp[2:] = np.asarray(target_shape)
    target_shape_tmp = tuple(target_shape_tmp)
    reshaped = torch.zeros(target_shape_tmp, dtype=seg.dtype, device=seg.device)
    unique_labels = torch.unique(seg, sorted=True)

    if processes is None:
        for i, label in enumerate(tqdm(unique_labels, desc=desc, disable=disable)):
            mask = seg == label
            reshaped_multihot = functional.interpolate(mask.float(), target_shape, mode='trilinear')
            reshaped[reshaped_multihot >= 0.5] = label
    else:
        reshaped_multihot_tensors = ptqdm(_resample_seg_smooth, unique_labels, processes, desc=desc, disable=disable, seg=seg, target_shape=target_shape)

        for i, label in enumerate(unique_labels):
            reshaped[reshaped_multihot_tensors[i] >= 0.5] = label

    return reshaped


def _resample_seg_smooth(label: torch.Tensor, seg: torch.Tensor, target_shape: Tuple[int, int, int]) -> torch.Tensor:
    """
    Resamples the given label tensor to the target shape.

    :param label: A tensor containing the label.
    :param seg: The segmentation tensor containing the label.
    :param target_shape: A tuple containing the target shape of the label tensor.
    :return: A tensor with the resampled label.
    """
    mask = seg == label
    reshaped_multihot = functional.interpolate(mask.float(), target_shape, mode='trilinear')
    return reshaped_multihot


def standardize(img: np.ndarray, zscore: dict) -> np.ndarray:
    """
    Standardizes the given image using the z-score normalization.

    :param img: A NumPy array containing the image.
    :param zscore: A dictionary containing the mean and standard deviation values used for z-score normalization.
    :return: A standardized image as a NumPy array.
    """
    img = img.astype(np.float32)
    img -= float(zscore["mean"])
    img /= float(zscore["std"])
    return img


def normalize(x: np.ndarray, x_min: float = None, x_max: float = None) -> np.ndarray:
    """
    Normalizes the given image within the range [0, 1] using the minimum and maximum values.

    :param x: A NumPy array containing the image.
    :param x_min: The minimum value of the image. If None, the minimum value of the image is used.
    :param x_max: The maximum value of the image. If None, the maximum value of the image is used.
    :return: A normalized image as a NumPy array.
    """
    if x_min is None:
        x_min = x.min()

    if x_max is None:
        x_max = x.max()

    if x_min == x_max:
        return x * 0
    else:
        return (x - x.min()) / (x.max() - x.min())


def load_filepaths(load_dir: str, extension: str = None, return_path: bool = True, return_extension: bool = True) -> np.ndarray:
    """
    Given a directory path, returns an array of file paths with the specified extension.

    Args:
        load_dir: The directory containing the files.
        extension: A string or list of strings specifying the file extension(s) to search for. Optional.
        return_path: If True, file paths will include the directory path. Optional.
        return_extension: If True, file paths will include the file extension. Optional.

    Returns:
        An array of file paths.
    """
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


def load_nifti(filename: str, return_meta: bool = False, is_seg: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[float, float, float], None, dict]]:
    """
    Load a NIfTI file and return it as a numpy array.

    Args:
        filename: The path to the NIfTI file.
        return_meta: If True, return the image metadata. Optional.
        is_seg: If True, round image values to nearest integer. Optional.

    Returns:
        The NIfTI file as a numpy array. If return_meta is True, a tuple with the image numpy array, the image
        spacing, affine transformation matrix and image metadata dictionary will be returned.
    """
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


def save_nifti(filename: str,
               image: np.ndarray,
               spacing: tuple[float] = None,
               affine: np.ndarray = None,
               header: dict = None,
               is_seg: bool = False,
               dtype: np.dtype = None) -> None:
    """
    Saves a NIfTI file to disk.

    Args:
        filename (str): The filename of the NIfTI file to save.
        image (np.ndarray): The image data to save.
        spacing (tuple[float], optional): The voxel spacing in mm. Defaults to None.
        affine (np.ndarray, optional): The affine transform matrix. Defaults to None.
        header (dict, optional): A dictionary of meta-data to save. Defaults to None.
        is_seg (bool, optional): Whether the image is a segmentation. If True, the image is rounded to the nearest
            integer and converted to int16 data type. Defaults to False.
        dtype (np.dtype, optional): The data type to save the image as. If None, the data type is determined by the
            image data. Defaults to None.

    Returns:
        None
    """
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

    sitk.WriteImage(image, filename)


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)


def get_identifiers_from_splitted_files(folder: str) -> List[str]:
    """
    Extracts unique identifier strings from a directory of files with '_<index>.nii.gz' suffixes.

    :param folder: The directory path to search.
    :return: A list of unique identifier strings.
    """
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def subfiles(folder: str, join: bool = True, prefix: str = None, suffix: str = None,
             sort: bool = True) -> List[str]:
    """
    Returns a list of files in a directory that match a specified prefix and/or suffix.

    :param folder: The directory path to search.
    :param join: Whether to join the directory path with the file names in the returned list (default True).
    :param prefix: The prefix that file names must have (optional).
    :param suffix: The suffix that file names must have (optional).
    :param sort: Whether to sort the returned list of file names (default True).
    :return: A list of file names.
    """
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isfile(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


def save_json(obj: dict, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    """
    Save a dictionary object to a JSON file.

    Args:
        obj (dict): A dictionary object to be saved.
        file (str): The file path where the dictionary object will be saved as JSON.
        indent (int): The number of spaces used to indent the output JSON file. Defaults to 4.
        sort_keys (bool): If True, sort the keys of the dictionary before writing it to the file. Defaults to True.

    Returns:
        None
    """
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)
