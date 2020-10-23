# This file should contain all the functions required by Wrapper.py

from pathlib import Path

import numpy as np
from skimage import io


def compute_hist(image_path: Path, num_bins: int) -> list:
    # bins_vec and freq_vec should contain values computed by custom function
    # bins_vec_lib and freq_vec_lib should contain values computed by python library function
    image_data = io.imread(image_path)

    # using inbuilt function
    freq_vec_lib, bins = np.histogram(
        image_data, bins=num_bins, range=(0, 255))
    bins_vec_lib = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]

    # using custom function
    bin_width = 255 / num_bins
    bins_vec = np.arange(bin_width / 2, 255, bin_width)
    binned_values = (image_data / bin_width).astype(int)
    freq_vec = [(binned_values == bin_number).sum()
                for bin_number in range(num_bins)]
    # adding all the pixels with value 255 to the last bin
    freq_vec[-1] += (binned_values == num_bins).sum()

    return [bins_vec, freq_vec, bins_vec_lib, freq_vec_lib]


def otsu_threshold(gray_image_path: Path) -> list:
    thr_w = thr_b = time_w = time_b = 0
    bin_image = None
    return [thr_w, thr_b, time_w, time_b, bin_image]


def change_background(quote_image_path: Path, bg_image_path: Path) -> np.ndarray:
    modified_image = None
    return modified_image


def count_connected_components(gray_image_path: Path) -> int:
    num_characters = 0
    return num_characters


def binary_morphology(gray_image_path: Path) -> np.ndarray:
    cleaned_image = None
    return cleaned_image


def count_mser_components(gray_image_path: Path) -> list:
    mser_binary_image = None
    otsu_binary_image = None
    num_mser_components = 0
    num_otsu_components = 0
    return [mser_binary_image, otsu_binary_image, num_mser_components, num_otsu_components]


# Testing code delete later
compute_hist('coins.png', 10)
