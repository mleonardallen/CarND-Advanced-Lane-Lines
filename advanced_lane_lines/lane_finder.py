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

    height = image.shape[0]
    width = image.shape[1]
    midpoint = int(width / 2)
    slices = 30
    search_window = 70

    # determine starting points for seeking
    left_pos = np.argmax(histogram[:midpoint])
    right_pos = np.argmax(histogram[midpoint:]) + midpoint

    # blank image to store lane line pixels
    left_blank = np.zeros_like(image)
    right_blank = np.zeros_like(image)
    # for debugging seek process
    rectangles = np.dstack((image, image, image))

    for bottom in range(height, 0, -slices):
        top = bottom - slices
        image_slice = image[top : bottom, :]

        # draw window for debugging
        cv2.rectangle(rectangles, (left_pos - search_window, bottom), (left_pos + search_window, top), (0,255,0), 2)
        cv2.rectangle(rectangles, (right_pos - search_window, bottom), (right_pos + search_window, top), (0,255,0), 2)

        left_blank, left_pos = seek(left_blank, image_slice, left_pos, top, bottom, search_window)
        right_blank, right_pos = seek(right_blank, image_slice, right_pos, top, bottom, search_window)

    # debugging image
    both = cv2.addWeighted(left_blank, 1, right_blank, 1, 0)
    both = np.dstack((both, both, both)) * 255
    both = cv2.addWeighted(both, 1, rectangles, 1, 0)
    Logger.save(both, 'lane-seek')

    return left_blank, right_blank

def seek(blank, image_slice, pos, top, bottom, search_window):
    pixels = image_slice[:, (pos - search_window): (pos + search_window)]
    blank[top : bottom, (pos - search_window): (pos + search_window)][pixels == 1] = 1

    if not np.count_nonzero(pixels):
        return blank, pos

    nonzero = pixels.nonzero()
    nonzerox = nonzero[1]
    pos += int(np.mean(nonzerox)) - search_window

    return blank, pos

def get_histogram(image):
    return np.sum(image[image.shape[0]/2:,:], axis=0)

def plot(left, right, shape):

    # save curvature plot
    fig = plt.figure(figsize=(8, 6))
    plt.xlim(0, shape[1])
    plt.ylim(0, shape[0])

    left.plot(color='red')
    right.plot(color='blue')

    plt.gca().invert_yaxis()
    Logger.save(fig, 'curvature')
    plt.close()
