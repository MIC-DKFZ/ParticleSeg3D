import sys
from pathlib import Path
sys.path.append(str(Path('').absolute().parent))

from skimage.measure import regionprops
from particleseg3d.utils import utils
from tqdm import tqdm
from os.path import join
import pickle
import numpy as np


def preprocess_seg_all(load_dir, metadata_filepath):
    names = utils.load_filepaths(load_dir, return_path=False, return_extension=False)
    metadata = {}

    len_props_total, len_props_filtered_total = 0, 0
    for name in tqdm(names):
        instance_seg = utils.load_nifti(join(load_dir, name + ".nii.gz"))
        props, len_props, len_props_filtered = preprocess_seg_single(instance_seg)
        len_props_total += len_props
        len_props_filtered_total += len_props_filtered
        metadata[name] = props

    with open(metadata_filepath, 'wb') as handle:
        pickle.dump(metadata, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("len props: {}, props_filtered: {}, ratio: {}".format(len_props_total, len_props_filtered_total, len_props_filtered_total / len_props_total))


def preprocess_seg_single(instance_seg):
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

    # print("len props: {}, props_filtered: {}".format(len(props), len(props_filtered)))
    return props_filtered, len(props), len(props_filtered)


if __name__ == '__main__':
    task_path = "/home/k539i/Documents/datasets/preprocessed/nnUNet/nnUNet_raw_data/nnUNet_raw_data/Task205_particle_seg/"
    load_dir = task_path + "labelsTr_instance"
    metadata_filepath = task_path + "regionprops.pkl"

    preprocess_seg_all(load_dir, metadata_filepath)
