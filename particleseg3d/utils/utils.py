from torch.nn import functional
from tqdm import tqdm
from acvl_utils.miscellaneous.ptqdm import ptqdm
import json
from typing import Tuple, List, Union
import importlib
import logging
import os
import shutil
import sys
import numpy as np
import torch
from torch import optim
import SimpleITK as sitk
from os.path import join
from natsort import natsorted


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


def save_checkpoint(state, is_best, checkpoint_dir):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


# def save_network_output(output_path, output, logger=None):
#     if logger is not None:
#         logger.info(f'Saving network output to: {output_path}...')
#     output = output.detach().cpu()[0]
#     with h5py.File(output_path, 'w') as f:
#         f.create_dataset('predictions', data=output, compression='gzip')


loggers = {}


def get_logger(name, level=logging.INFO):
    global loggers
    if loggers.get(name) is not None:
        return loggers[name]
    else:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        # Logging to console
        stream_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        loggers[name] = logger

        return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        logger.info(f"Current patch size: {shape}")
        model(patch)


def remove_halo(patch, index, shape, patch_halo):
    """
    Remove `pad_width` voxels around the edges of a given patch.
    """
    assert len(patch_halo) == 3

    def _new_slices(slicing, max_size, pad):
        if slicing.start == 0:
            p_start = 0
            i_start = 0
        else:
            p_start = pad
            i_start = slicing.start + pad

        if slicing.stop == max_size:
            p_stop = None
            i_stop = max_size
        else:
            p_stop = -pad if pad != 0 else 1
            i_stop = slicing.stop - pad

        return slice(p_start, p_stop), slice(i_start, i_stop)

    D, H, W = shape

    i_c, i_z, i_y, i_x = index
    p_c = slice(0, patch.shape[0])

    p_z, i_z = _new_slices(i_z, D, patch_halo[0])
    p_y, i_y = _new_slices(i_y, H, patch_halo[1])
    p_x, i_x = _new_slices(i_x, W, patch_halo[2])

    patch_index = (p_c, p_z, p_y, p_x)
    index = (i_c, i_z, i_y, i_x)
    return patch[patch_index], index


def number_of_features_per_level(init_channel_number, num_levels):
    return [init_channel_number * 2 ** k for k in range(num_levels)]


class _TensorboardFormatter:
    """
    Tensorboard formatters converts a given batch of images (be it input/output to the network or the target segmentation
    image) to a series of images that can be displayed in tensorboard. This is the parent class for all tensorboard
    formatters which ensures that returned images are in the 'CHW' format.
    """

    def __init__(self, **kwargs):
        pass

    def __call__(self, name, batch):
        """
        Transform a batch to a series of tuples of the form (tag, img), where `tag` corresponds to the image tag
        and `img` is the image itself.

        Args:
             name (str): one of 'inputs'/'targets'/'predictions'
             batch (torch.tensor): 4D or 5D torch tensor
        """

        def _check_img(tag_img):
            tag, img = tag_img

            assert img.ndim == 2 or img.ndim == 3, 'Only 2D (HW) and 3D (CHW) images are accepted for display'

            if img.ndim == 2:
                img = np.expand_dims(img, axis=0)
            else:
                C = img.shape[0]
                assert C == 1 or C == 3, 'Only (1, H, W) or (3, H, W) images are supported'

            return tag, img

        tagged_images = self.process_batch(name, batch)

        return list(map(_check_img, tagged_images))

    def process_batch(self, name, batch):
        raise NotImplementedError


class DefaultTensorboardFormatter(_TensorboardFormatter):
    def __init__(self, skip_last_target=False, **kwargs):
        super().__init__(**kwargs)
        self.skip_last_target = skip_last_target

    def process_batch(self, name, batch):
        if name == 'targets' and self.skip_last_target:
            batch = batch[:, :-1, ...]

        tag_template = '{}/batch_{}/channel_{}/slice_{}'

        tagged_images = []

        if batch.ndim == 5:
            # NCDHW
            slice_idx = batch.shape[2] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                for channel_idx in range(batch.shape[1]):
                    tag = tag_template.format(name, batch_idx, channel_idx, slice_idx)
                    img = batch[batch_idx, channel_idx, slice_idx, ...]
                    tagged_images.append((tag, self._normalize_img(img)))
        else:
            # batch has no channel dim: NDHW
            slice_idx = batch.shape[1] // 2  # get the middle slice
            for batch_idx in range(batch.shape[0]):
                tag = tag_template.format(name, batch_idx, 0, slice_idx)
                img = batch[batch_idx, slice_idx, ...]
                tagged_images.append((tag, self._normalize_img(img)))

        return tagged_images

    @staticmethod
    def _normalize_img(img):
        return np.nan_to_num((img - np.min(img)) / np.ptp(img))


def _find_masks(batch, min_size=10):
    """Center the z-slice in the 'middle' of a given instance, given a batch of instances

    Args:
        batch (ndarray): 5d numpy tensor (NCDHW)
    """
    result = []
    for b in batch:
        assert b.shape[0] == 1
        patch = b[0]
        z_sum = patch.sum(axis=(1, 2))
        coords = np.where(z_sum > min_size)[0]
        if len(coords) > 0:
            ind = coords[len(coords) // 2]
            result.append(b[:, ind:ind + 1, ...])
        else:
            ind = b.shape[1] // 2
            result.append(b[:, ind:ind + 1, ...])

    return np.stack(result, axis=0)


def get_tensorboard_formatter(formatter_config):
    if formatter_config is None:
        return DefaultTensorboardFormatter()

    class_name = formatter_config['name']
    m = importlib.import_module('pytorch3dunet.unet3d.utils')
    clazz = getattr(m, class_name)
    return clazz(**formatter_config)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to its corresponding one-hot vector.
    It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert input.dim() == 4

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def convert_to_numpy(*inputs):
    """
    Coverts input tensors to numpy ndarrays

    Args:
        inputs (iteable of torch.Tensor): torch tensor

    Returns:
        tuple of ndarrays
    """

    def _to_numpy(i):
        assert isinstance(i, torch.Tensor), "Expected input to be torch.Tensor"
        return i.detach().cpu().numpy()

    return (_to_numpy(i) for i in inputs)


def create_optimizer(optimizer_config, model):
    learning_rate = optimizer_config['learning_rate']
    weight_decay = optimizer_config.get('weight_decay', 0)
    betas = tuple(optimizer_config.get('betas', (0.9, 0.999)))
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=betas, weight_decay=weight_decay)
    return optimizer


def create_lr_scheduler(lr_config, optimizer):
    if lr_config is None:
        return None
    class_name = lr_config.pop('name')
    m = importlib.import_module('torch.optim.lr_scheduler')
    clazz = getattr(m, class_name)
    # add optimizer to the config
    lr_config['optimizer'] = optimizer
    return clazz(**lr_config)


def get_class(class_name, modules):
    for module in modules:
        m = importlib.import_module(module)
        clazz = getattr(m, class_name, None)
        if clazz is not None:
            return clazz
    raise RuntimeError(f'Unsupported dataset class: {class_name}')


def get_model(config):
    model_class = get_class(config['model']['name'], modules=['model'])
    return model_class(config, **config['model'])


def get_model_class(config):
    model_class = get_class(config['model']['name'], modules=['model'])
    return model_class


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
               spacing: Tuple[float] = None,
               affine: np.ndarray = None,
               header: dict = None,
               is_seg: bool = False,
               dtype: np.dtype = None) -> None:
    """
    Saves a NIfTI file to disk.

    Args:
        filename (str): The filename of the NIfTI file to save.
        image (np.ndarray): The image data to save.
        spacing (Tuple[float], optional): The voxel spacing in mm. Defaults to None.
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


def pixel2mm(length: Tuple[float], spacing: Tuple[float]) -> np.ndarray:
    """
    Convert a length in pixel to millimeter using the given spacing.

    Args:
        length (Tuple[float]): Length in pixel.
        spacing (Tuple[float]): Voxel spacing in millimeter.

    Returns:
        np.ndarray: Length in millimeter.
    """
    return np.asarray(length) * np.asarray(spacing)


def mm2pixel(length: Tuple[float], spacing: Tuple[float]) -> np.ndarray:
    """
    Convert a length in millimeter to pixel using the given spacing.

    Args:
        length (Tuple[float]): Length in millimeter.
        spacing (Tuple[float]): Voxel spacing in millimeter.

    Returns:
        np.ndarray: Length in pixel.
    """
    return np.asarray(length) / np.asarray(spacing)


def compute_size_conversion_factor(source_particle_size_in_mm: Tuple[float], source_spacing: Tuple[float],
                                    target_particle_size_in_mm: Tuple[float], target_spacing: Tuple[float]) -> np.ndarray:
    """
    Compute the conversion factor between the source and target size in pixel.

    Args:
        source_particle_size_in_mm (Tuple[float]): Particle size of the source image in millimeter.
        source_spacing (Tuple[float]): Voxel spacing of the source image in millimeter.
        target_particle_size_in_mm (Tuple[float]): Particle size of the target image in millimeter.
        target_spacing (Tuple[float]): Voxel spacing of the target image in millimeter.

    Returns:
        np.ndarray: Conversion factor.
    """
    factor = np.asarray(target_spacing) / np.asarray(source_spacing)
    factor *= np.asarray(source_particle_size_in_mm) / np.asarray(target_particle_size_in_mm)
    return factor
