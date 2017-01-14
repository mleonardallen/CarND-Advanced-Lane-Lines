from advanced_lane_lines.log import Logger
from configparser import ConfigParser
import numpy as np
import cv2
import matplotlib.pyplot as plt

config = ConfigParser()
config.read('config.cfg')
thresholds = config.items('thresholds')
thresholds = {item[0]: tuple(map(float, item[1].split())) for item in thresholds}

def combined_thresh(image):
    """ Returns thresholded binary image
    Sobel directional, magnitude, and direction combined
    """

    # Sobel threshold
    mag_binary = mag_thresh(image, sobel_kernel=3, thresh=thresholds.get('magnitude'))
    dir_binary = dir_threshold(image, sobel_kernel=3, thresh=thresholds.get('direction'))

    sobel_binary = np.zeros_like(dir_binary)
    sobel_binary[((mag_binary == 1) & (dir_binary == 1))] = 1

    Logger.save(sobel_binary, 'magnitude-and-direction')

    # Saturation threshold
    hls_binary = hls_select(image, thresh=thresholds.get('saturation_high'))

    binary_output = np.zeros_like(image)
    binary_output[(sobel_binary == 1) | (hls_binary == 1)] = 1

    Logger.save(binary_output, 'combined')

    return binary_output

def hls_select(img, thresh=(0, 1)):
    """ Returns HLS channel thresholded binary image """

    # Convert to HLS color space
    saturation_channel = 2
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img = img[:,:,saturation_channel]

    # Apply a threshold to the saturation channel
    binary_output = np.zeros_like(img)
    binary_output[(img > thresh[0]) & (img <= thresh[1])] = 1

    Logger.save(binary_output, 'hls')

    return binary_output

def abs_sobel_thresh(img, orient = 'x', sobel_kernel = 3, thresh = (0, 255)):
    """ Sobel directional gradient thresholded binary image """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Take the gradient in given orientation
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)

    # Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a binary mask where thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    Logger.save(binary_output, 'sobel' + orient)

    return binary_output

def mag_thresh(img, sobel_kernel = 3, thresh = (0, 255)):
    """ Sobel magnitude thresholded binary image
    Define a function that applies Sobel x and y, 
    then computes the magnitude of the gradient
    and applies a threshold
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # Calculate the magnitude 
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    Logger.save(binary_output, 'magnitude')

    return binary_output

def dir_threshold(img, sobel_kernel = 3, thresh = (0, np.pi / 2)):
    """ Sobel direction thresholded binary image
    Define a function that applies Sobel x and y, 
    then computes the direction of the gradient
    and applies a threshold.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    arctan_sobel = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(arctan_sobel)
    binary_output[(arctan_sobel >= thresh[0]) & (arctan_sobel <= thresh[1])] = 1

    Logger.save(binary_output, 'direction')

    return binary_output
