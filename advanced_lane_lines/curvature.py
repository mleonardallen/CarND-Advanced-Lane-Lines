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

    fig = plt.figure(figsize=(8, 6))
    fit_line(left_blank, color='red')
    fit_line(right_blank, color='blue')
    plt.gca().invert_yaxis()
    Logger.save(fig, 'curvature')
    plt.close()


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

    # Logger.save(blank, 'slices')

def fit_line(image, color='red'):

    yvals, xvals = image.nonzero()
    fit = np.polyfit(yvals, xvals, 2)
    fitx = fit[0] * yvals ** 2 + fit[1] * yvals + fit[2]

    plt.plot(xvals, yvals, 'o', color=color)
    plt.plot(fitx, yvals, color='green', linewidth=3)

    curvature = get_curvature(yvals, fit)

def get_curvature_radians():
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension

    fit_cr = np.polyfit(yvals * ym_per_pix, xvals * xm_per_pix, 2)
    curverad = ((1 + (2 * fit_cr[0] * y_eval + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0])
    # Now our radius of curvature is in meters
    # Example values: 3380.7 m    3189.3 m

def get_curvature(yvals, fit):
    # Define y-value where we want radius of curvature
    y_eval = np.max(yvals)
    curverad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2*fit[0])
    return curverad
    # Example values: 1163.9    1213.7

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