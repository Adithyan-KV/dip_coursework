# This file should contain all the functions required by Wrapper.py

from pathlib import Path

import numpy as np
from skimage import io
import time
import matplotlib.pyplot as plt


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
    image_data = io.imread(gray_image_path)
    total_pixels = image_data.size

    timestamp_1 = time.time()

    # Threshold by minimizing within class variance
    within_class_variances = np.zeros(256)
    for threshold in range(256):
        freq_c1 = (image_data <= threshold).sum()
        freq_c2 = total_pixels - freq_c1
        # calculating weights
        w1 = freq_c1 / total_pixels
        w2 = 1 - w1
        # handles division by zero for classes with zero elements
        if freq_c1 != 0 and freq_c2 != 0:
            # calculating variances for each class
            v1 = np.var(image_data[image_data <= threshold])
            v2 = np.var(image_data[image_data > threshold])
            within_class_variances[threshold] = v1 * w1 + v2 * w2
        else:
            # if one class is empty, within class var is total image var
            within_class_variances[threshold] = np.var(image_data)
    thr_w = np.argmin(within_class_variances)

    timestamp_2 = time.time()

    # Threshold by maximizing between class variance
    between_class_variances = np.zeros(256)
    for threshold in range(256):
        freq_c1 = (image_data <= threshold).sum()
        freq_c2 = total_pixels - freq_c1
        # calculating weights
        w1 = freq_c1 / total_pixels
        w2 = 1 - w1
        # means
        if freq_c1 != 0 and freq_c2 != 0:
            m1 = np.mean((image_data[image_data <= threshold]))
            m2 = np.mean((image_data[image_data > threshold]))
            between_class_variances[threshold] = w1 * w2 * ((m1 - m2)**2)
        else:
            between_class_variances[threshold] = 0
    thr_b = np.argmax(between_class_variances)

    timestamp_3 = time.time()

    time_w = timestamp_2 - timestamp_1
    time_b = timestamp_3 - timestamp_2

    bin_image = (image_data > thr_b) * 255
    return [thr_w, thr_b, time_w, time_b, bin_image]


def change_background(quote_image_path: Path, bg_image_path: Path) -> np.ndarray:
    background_data = io.imread(bg_image_path)
    quote_data = io.imread(quote_image_path)

    binarized_quote = otsu_threshold(quote_image_path)[4]
    mask = (binarized_quote > 0)

    # using binarized image as a mask to combine fg and bg
    modified_image = np.invert(mask) * quote_data + mask * background_data
    return modified_image


def count_connected_components(gray_image_path: Path) -> int:
    bin_img = otsu_threshold(gray_image_path)[4]
    bin_img = (bin_img / 255).astype(int)
    print(bin_img)
    label_matrix = np.zeros_like(bin_img)

    k = 1
    rows, columns = bin_img.shape
    for i in range(rows):
        for j in range(columns):
            if bin_img[i, j] == 0:
                # corner
                if i == 0 and j == 0:
                    label_matrix[i, j] = k
                    k += 1

                # top edge
                elif i == 0:
                    if bin_img[i, j - 1] == 1:
                        label_matrix[i, j] = k
                        k += 1

                # left edge
                elif j == 0:
                    if bin_img[i - 1, j] == 1:
                        label_matrix[i, j] = k
                        k += 1

                # general case
                else:
                    left_label = label_matrix[i, j - 1]
                    top_label = label_matrix[i - 1, j]
                    # if both top and left have label
                    if bin_img[i, j - 1] == 0 and bin_img[i - 1, j] == 0:
                        if left_label == top_label:
                            label_matrix[i, j] = top_label
                        else:
                            label_matrix[i, j] = top_label
                            # replace all previous left labels to top label
                            label_matrix[label_matrix ==
                                         left_label] = top_label
                    # if only one of top or left has label, copy label
                    elif bin_img[i, j - 1] == 0 and bin_img[i - 1, j] == 1:
                        label_matrix[i, j] == left_label
                    elif bin_img[i - 1, j] == 0 and bin_img[i, j - 1] == 1:
                        label_matrix[i, j] == top_label
                    # if both are unlabelled
                    elif bin_img[i, j - 1] == 1 and bin_img[i - 1, j] == 1:
                        label_matrix[i, j] = k
                        k += 1

    num = np.unique(label_matrix)
    # print(label_matrix)
    # print(num)
    print(len(num))
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
count_connected_components('quote.png')
