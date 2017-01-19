from advanced_lane_lines.log import Logger
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import KMeans

def histogram(image):

    histogram = get_histogram(image)
    # save picture of histogram
    fig = plt.figure(figsize=(8, 6))
    plt.plot(histogram)
    Logger.save(fig, 'histogram')

    peaks = get_peaks(histogram)
    height = image.shape[0]
    slices = 30
    search_window = 50

    left_pos = peaks[0]
    right_pos = peaks[-1]

    # blank image to store lane line pixels
    blank = np.zeros_like(image)

    for bottom in range(height, 0, -slices):
        top = bottom - slices
        image_slice = image[top : bottom, :]

        # left window
        left_pixels = image_slice[:, (left_pos - search_window): (left_pos + search_window)]
        blank[top : bottom, (left_pos - search_window): (left_pos + search_window)][left_pixels == 1] = 1
        peaks = get_peaks(get_histogram(left_pixels))
        if len(peaks):
            left_pos += peaks[0] - search_window

        right_pixels = image_slice[:, (right_pos - search_window): (right_pos + search_window)]
        blank[top : bottom, (right_pos - search_window): (right_pos + search_window)][right_pixels == 1] = 1

        peaks = get_peaks(get_histogram(right_pixels))
        if len(peaks):
            right_pos += peaks[0] - search_window

    Logger.save(blank, 'slices')

def get_peaks(histogram):
    if not np.count_nonzero(histogram):
        return []
    return signal.find_peaks_cwt(histogram, np.arange(30,300))

def get_histogram(image):
    return np.sum(image[image.shape[0]/2:,:], axis=0)

def scatterplot():
    # Generate some fake data to represent lane-line pixels
    yvals = np.linspace(0, 100, num=101)*7.2  # to cover same y-range as image
    leftx = np.array([200 + (elem**2)*4e-4 + np.random.randint(-50, high=51) 
                                  for idx, elem in enumerate(yvals)])
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = np.array([900 + (elem**2)*4e-4 + np.random.randint(-50, high=51) 
                                    for idx, elem in enumerate(yvals)])
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    # Fit a second order polynomial to each fake lane line
    left_fit = np.polyfit(yvals, leftx, 2)
    left_fitx = left_fit[0]*yvals**2 + left_fit[1]*yvals + left_fit[2]
    right_fit = np.polyfit(yvals, rightx, 2)
    right_fitx = right_fit[0]*yvals**2 + right_fit[1]*yvals + right_fit[2]

    # Plot up the fake data
    plt.plot(leftx, yvals, 'o', color='red')
    plt.plot(rightx, yvals, 'o', color='blue')
    plt.xlim(0, 1280)
    plt.ylim(0, 720)
    plt.plot(left_fitx, yvals, color='green', linewidth=3)
    plt.plot(right_fitx, yvals, color='green', linewidth=3)
    plt.gca().invert_yaxis() # to visualize as we do the images