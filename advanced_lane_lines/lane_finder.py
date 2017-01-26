from advanced_lane_lines.log import Logger
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import cv2

def get_lane_pixels(image):

    histogram = get_histogram(image)

    # save picture of histogram
    fig = plt.figure(figsize=(8, 6))
    plt.plot(histogram)
    Logger.save(fig, 'histogram')
    plt.close()

    peaks = get_peaks(histogram)
    height = image.shape[0]
    slices = 30
    search_window = 50

    left_pos = peaks[0]
    right_pos = peaks[-1]

    # blank image to store lane line pixels
    left_blank = np.zeros_like(image)
    right_blank = np.zeros_like(image)

    for bottom in range(height, 0, -slices):
        top = bottom - slices
        image_slice = image[top : bottom, :]

        left_blank, left_pos = seek(left_blank, image_slice, left_pos, top, bottom, search_window)
        right_blank, right_pos = seek(right_blank, image_slice, right_pos, top, bottom, search_window)

    return left_blank, right_blank

def seek(blank, image_slice, pos, top, bottom, search_window):
    pixels = image_slice[:, (pos - search_window): (pos + search_window)]
    blank[top : bottom, (pos - search_window): (pos + search_window)][pixels == 1] = 1

    peaks = get_peaks(get_histogram(pixels))
    if len(peaks):
        pos += peaks[0] - search_window

    return blank, pos

def get_peaks(histogram):
    if not np.count_nonzero(histogram):
        return []
    return signal.find_peaks_cwt(histogram, np.arange(30,300))

def get_histogram(image):
    return np.sum(image[image.shape[0]/2:,:], axis=0)

def plot(left, right, shape):

    # save curvature plot
    fig = plt.figure(figsize=(8, 6))
    plt.xlim(0, shape[1])
    plt.ylim(0, shape[0])

    left.plot()
    right.plot()

    plt.gca().invert_yaxis()
    Logger.save(fig, 'curvature')
    plt.close()
