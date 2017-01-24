from advanced_lane_lines.log import Logger
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from sklearn.cluster import KMeans
import cv2

def get_curvature_values(image):

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
    left_yvals, left_fitx, left_curvature = fit_line(left_blank, color='red')
    right_yvals, right_fitx, right_curvature = fit_line(right_blank, color='blue')
    plt.xlim(0, image.shape[1])
    plt.ylim(0, image.shape[0])
    plt.gca().invert_yaxis()
    Logger.save(fig, 'curvature')
    plt.close()

    return left_yvals, right_yvals, left_fitx, right_fitx

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
    return yvals, fitx, curvature

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
    # Example values: 1163.9    1213.7
    return curverad

def draw(image, left_yvals, right_yvals, left_fitx, right_fitx):

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, left_yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the blank image
    image = cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    return image

