from advanced_lane_lines.log import Logger
import advanced_lane_lines.util as util
from configparser import ConfigParser
import numpy as np
import cv2
import matplotlib.pyplot as plt

config = ConfigParser()
config.read('config.cfg')
thresholds = util.get_config_tuples('thresholds')

kernels = config.items('threshold_kernels')
kernels = {item[0]: int(item[1]) for item in kernels}

def combined_thresh(image):
    """ Returns thresholded binary image
    Sobel directional, magnitude, and direction combined
    """

    sobelx = abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=thresholds.get('sobelx'))
    Logger.save(sobelx, 'sobelx')

    # Sobel threshold
    # mag_binary = mag_thresh(image, sobel_kernel=kernels.get('magnitude'), thresh=thresholds.get('magnitude'))
    # Logger.save(mag_binary, 'magnitude-binary')

    # dir_binary = dir_threshold(image, sobel_kernel=kernels.get('direction'), thresh=thresholds.get('direction'))
    # Logger.save(dir_binary, 'direction-binary')

    # sobel_binary = np.zeros_like(mag_binary)
    # sobel_binary[((mag_binary == 1) & (dir_binary == 1))] = 1
    # Logger.save(sobel_binary, 'magnitude-and-direction-binary')

    # Saturation threshold (high)
    saturation_binary = hls_select(image, thresh=thresholds.get('saturation'), channel=2)
    Logger.save(saturation_binary, 'saturation-binary')

    lightness_binary = hls_select(image, thresh=thresholds.get('lightness'), channel=1)
    Logger.save(lightness_binary, 'lightness-binary')

    sat_and_light = np.zeros_like(saturation_binary)
    sat_and_light[((saturation_binary == 1) & (lightness_binary == 1))] = 1
    Logger.save(sat_and_light, 'saturation-and-lightness')

    binary_output = np.zeros(image.shape[:2], dtype='uint8')
    binary_output[(sobelx == 1) | (sat_and_light == 1)] = 1

    return binary_output

def hls_select(img, thresh=(0, 1), channel=2):
    """ Returns HLS channel thresholded binary image """

    # Convert to HLS color space

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img = img[:,:, channel]

    # Apply a threshold to the saturation channel
    binary_output = np.zeros_like(img, dtype='uint8')
    binary_output[(img > thresh[0]) & (img <= thresh[1])] = 1

    return binary_output

def abs_sobel_thresh(img, orient = 'x', sobel_kernel = 3, thresh = (0, 255)):
    """ Sobel directional gradient thresholded binary image """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # kernel = np.ones((3, 3), np.uint8)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    # Logger.save(gray, 'blur')

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
    binary_output = np.zeros_like(scaled_sobel, dtype='uint8')
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output

def mag_thresh(img, sobel_kernel = 3, thresh = (0, 255)):
    """ Sobel magnitude thresholded binary image
    Define a function that applies Sobel x and y, 
    then computes the magnitude of the gradient
    and applies a threshold
    """

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # kernel = np.ones((3, 3), np.uint8)
    # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # Calculate the magnitude 
    magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255*magnitude/np.max(magnitude))
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel, dtype='uint8')
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

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
    binary_output = np.zeros_like(arctan_sobel, dtype='uint8')
    binary_output[(arctan_sobel >= thresh[0]) & (arctan_sobel <= thresh[1])] = 1

    return binary_output
