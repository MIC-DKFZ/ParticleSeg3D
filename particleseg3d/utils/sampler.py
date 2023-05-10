import random
import torchio as tio
import numpy as np


class GridSampler:
    def __init__(self, subject, patch_size):
        self.subject = subject
        self.patch_size = patch_size
        self.init()

    def __iter__(self):
        self.init()
        return self

    def init(self):
        self.sampler = tio.GridSampler(subject=self.subject, patch_size=self.patch_size)
        self.length = len(self.sampler)
        self.sampler = self.sampler(self.subject)

    def __next__(self):
        return next(self.sampler)

    def __len__(self):
        return self.length


class MultiSizeUniformSampler:
    def __init__(self, subject, min_patch_size, max_patch_size, num_patches, isotropic=True, filter_empty=True):
        self.subject = subject
        self.subject_size = self.subject.shape[1:]
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.num_patches = num_patches
        self.isotropic = isotropic
        self.filter_empty = filter_empty
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.num_patches <= self.index:
            raise StopIteration
        patch = self.extract()
        if self.filter_empty and np.sum(np.nonzero(patch["image"])) <= 5:
            patch = self.extract()
        self.index += 1
        return patch

    def __len__(self):
        return self.num_patches

    def random_patch_size(self):
        if self.isotropic:
            index = np.argmin(np.abs(np.asarray(self.min_patch_size) - np.asarray(self.max_patch_size)))
            size = random.randint(self.min_patch_size[index], self.max_patch_size[index])
            return size, size, size
        else:
            w = random.randint(self.min_patch_size[0], self.max_patch_size[0])
            h = random.randint(self.min_patch_size[1], self.max_patch_size[1])
            d = random.randint(self.min_patch_size[2], self.max_patch_size[2])
            return w, h, d

    def random_position(self, patch_size):
        w = random.randint(0, self.subject_size[0] - patch_size[0])
        h = random.randint(0, self.subject_size[1] - patch_size[1])
        d = random.randint(0, self.subject_size[2] - patch_size[2])
        return w, h, d

    def cropping(self, position, patch_size):
        w_ini = position[0]
        w_fin = self.subject_size[0] - (position[0] + patch_size[0])
        h_ini = position[1]
        h_fin = self.subject_size[1] - (position[1] + patch_size[1])
        d_ini = position[2]
        d_fin = self.subject_size[2] - (position[2] + patch_size[2])
        return w_ini, w_fin, h_ini, h_fin, d_ini, d_fin

    def extract(self):
        patch_size = self.random_patch_size()
        # print("patch_size: ", patch_size)
        position = self.random_position(patch_size)
        cropping = self.cropping(position, patch_size)
        transform = tio.transforms.Crop(cropping=cropping)
        patch = transform(self.subject)
        patch["location"] = cropping
        return patch
