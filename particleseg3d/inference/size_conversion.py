import numpy as np


def compute_patch_size(target_spacing, target_particle_size_in_mm, target_patch_size_in_pixel, source_spacing, source_particle_size_in_mm, image_shape):
    size_conversion_factor = compute_size_conversion_factor(source_particle_size_in_mm, source_spacing, target_particle_size_in_mm, target_spacing)
    size_conversion_factor = np.around(size_conversion_factor, decimals=3)
    source_patch_size_in_pixel = np.rint(target_patch_size_in_pixel * size_conversion_factor).astype(int)

    # if image_shape[0] < source_patch_size_in_pixel[0] or image_shape[1] < source_patch_size_in_pixel[1] or image_shape[2] < source_patch_size_in_pixel[2]:
    #     max_index = np.argmax(np.asarray(source_patch_size_in_pixel) - np.asarray(image_shape))
    #     max_target_patch_size_in_pixel = (image_shape[max_index] / source_patch_size_in_pixel[max_index]) * np.asarray(target_patch_size_in_pixel)
    #     max_target_patch_size_in_pixel = np.floor(max_target_patch_size_in_pixel).astype(int)
    #     target_patch_size_in_pixel_old = target_patch_size_in_pixel
    #     source_patch_size_in_pixel_old = source_patch_size_in_pixel
    #     target_patch_size_in_pixel = tuple([int(value) for value in max_target_patch_size_in_pixel])
    #     source_patch_size_in_pixel = np.rint(target_patch_size_in_pixel * size_conversion_factor).astype(int)

    return target_patch_size_in_pixel, source_patch_size_in_pixel, size_conversion_factor


def pixel2mm(length, spacing):
    return np.asarray(length) * np.asarray(spacing)


def mm2pixel(length, spacing):
    return np.asarray(length) / np.asarray(spacing)


def compute_size_conversion_factor(source_particle_size_in_mm, source_spacing, target_particle_size_in_mm, target_spacing):
    factor = np.asarray(target_spacing) / np.asarray(source_spacing)
    factor *= np.asarray(source_particle_size_in_mm) / np.asarray(target_particle_size_in_mm)
    return factor